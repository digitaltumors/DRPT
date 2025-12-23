import os
import gc
import copy
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np

from src.utils.data import move_to


class DrugResponseTrainer(object):

    def __init__(self, drug_response_model, drug_response_dataloader_drug, drug_response_dataloader_cellline,
                 device, args, validation_dataloader=None, fix_embedding=False):
        self.device = device
        self.drug_response_model = drug_response_model.to(self.device)
        self.drug_response_dataloader_drug = drug_response_dataloader_drug
        self.drug_response_dataloader_cellline = drug_response_dataloader_cellline

        # --- Keep ONLY the drug response loss ---
        self.beta = 0.1  # retained for SmoothL1Loss(beta)
        self.drug_response_loss = nn.SmoothL1Loss(self.beta)

        # --- Optimizer ---
        self.args = args
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.drug_response_model.parameters()),
            lr=self.args.lr, weight_decay=self.args.wd
        )

        # --- Warmup + Cosine using LambdaLR (works on older torch) ---
        warmup_pct          = getattr(self.args, "warmup_pct", 0.03)          # ~3% of total steps
        warmup_start_factor = getattr(self.args, "warmup_start_factor", 0.1)  # start at 10% of target lr
        min_lr              = getattr(self.args, "min_lr", self.args.lr / 100)

        steps_per_epoch = len(self.drug_response_dataloader_drug)
        total_steps     = max(1, steps_per_epoch * self.args.epochs)
        warmup_steps    = max(1, int(total_steps * warmup_pct))
        cosine_steps    = max(1, total_steps - warmup_steps)

        base_lr = float(self.args.lr)
        min_factor = float(min_lr) / max(base_lr, 1e-12)  # scale factor at cosine end

        def lr_lambda(current_step: int):
            # Linear warmup from warmup_start_factor -> 1.0
            if current_step < warmup_steps:
                if warmup_steps == 0:
                    return 1.0
                progress = current_step / float(max(1, warmup_steps))
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress
            # Cosine decay from 1.0 -> min_factor
            progress = (current_step - warmup_steps) / float(max(1, cosine_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Data / masks
        self.validation_dataloader = validation_dataloader
        self.l2_lambda = self.args.l2_lambda
        self.total_train_step = len(self.drug_response_dataloader_drug) * self.args.epochs

        tp = self.drug_response_dataloader_drug.dataset.tree_parser
        self.nested_subtrees_forward = move_to(tp.get_nested_subtree_mask(self.args.subtree_order, direction='forward'), device)
        self.nested_subtrees_backward = move_to(tp.get_nested_subtree_mask(self.args.subtree_order, direction='backward'), device)
        self.gene2system_mask = move_to(torch.tensor(tp.gene2sys_mask, dtype=torch.bool), device)
        self.system2gene_mask = move_to(torch.tensor(tp.sys2gene_mask, dtype=torch.bool), device)
        print("%d sys2gene in Dataloader" % tp.sys2gene_mask.sum())

        # Book-keeping
        self.best_model = self.drug_response_model  # backward-compat fallback
        self.fix_embedding = fix_embedding
        self.g2p_module_names = ["Mut2Sys", "Sys2Cell", "Cell2Sys"]  # preserved for compatibility
        self.performance = {}  # {epoch: {..., "val_loss": float, "val_loss_per_drug": {...}, "lr": float}}
        self.loss = {}         # {epoch: mean_train_loss}

        # --- Track both best-by-EMA(val loss) and best-by-Pearson(mean per drug) ---
        self.best_model_by_ema = None
        self.best_ema = float("inf")
        self.best_model_by_pearson = None
        self.best_pearson = -float("inf")

        # Early stopping config (NOW defaults to False unless flag is set)
        self.use_early_stop = bool(getattr(self.args, "early_stop", False))
        self.es_beta      = getattr(self.args, "early_stop_ema_beta", 0.9)
        self.es_patience  = getattr(self.args, "early_stop_patience", 5)     # # of validation checks
        self.es_min_delta = getattr(self.args, "early_stop_min_delta", 0.0)  # require this much improvement
        self._ema_val = None
        self._bad_checks = 0

        # Optional: keep embeddings fixed during training
        if fix_embedding:
            if self.args.multiprocessing_distributed:
                self.system_embedding = copy.deepcopy(self.drug_response_model.module.system_embedding)
                self.gene_embedding = copy.deepcopy(self.drug_response_model.module.gene_embedding)
            else:
                self.system_embedding = copy.deepcopy(self.drug_response_model.system_embedding)
                self.gene_embedding = copy.deepcopy(self.drug_response_model.gene_embedding)

    def train(self, epochs, output_path=None):

        stop_training = False

        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            gc.collect()
            torch.cuda.empty_cache()

            # Validate on schedule (skipped if no val dataloader)
            if (epoch % self.args.val_step) == 0 and (self.validation_dataloader is not None):
                # log current LR (version-proof)
                current_lr = float(self.optimizer.param_groups[0]["lr"])
                print(f"[Validation] Epoch {epoch}: current LR = {current_lr:.6g}")

                # evaluate() fills self.performance[epoch] with val_loss, per-drug stats
                mean_pearson_per_drug = self.evaluate(self.drug_response_model, self.validation_dataloader, epoch, name="Validation")

                # attach LR into metrics for this epoch
                self.performance.setdefault(epoch, {})["lr"] = current_lr

                # --- Track best-by-Pearson (mean per-drug) ---
                if mean_pearson_per_drug > self.best_pearson:
                    self.best_pearson = mean_pearson_per_drug
                    self.best_model_by_pearson = copy.deepcopy(self.drug_response_model).to('cpu')
                    print(f"[Best-Pearson] New best mean per-drug Pearson {self.best_pearson:.6g} at epoch {epoch}")

                # --- Early stopping on EMA(val_loss) + best-by-EMA checkpoint ---
                if self.use_early_stop:
                    val_loss = float(self.performance[epoch]["val_loss"])
                    if self._ema_val is None:
                        self._ema_val = val_loss
                    else:
                        self._ema_val = self.es_beta * self._ema_val + (1.0 - self.es_beta) * val_loss

                    # Record EMA for plotting
                    self.performance[epoch]["val_loss_ema"] = float(self._ema_val)

                    improved = (self._ema_val < (self.best_ema - self.es_min_delta))
                    if improved:
                        self.best_ema = float(self._ema_val)
                        self._bad_checks = 0
                        self.best_model_by_ema = copy.deepcopy(self.drug_response_model).to('cpu')
                        print(f"[Best-EMA] New best EMA {self.best_ema:.6g} at epoch {epoch}")
                    else:
                        self._bad_checks += 1
                        print(f"[EarlyStop] No EMA improvement for {self._bad_checks}/{self.es_patience} validations "
                              f"(EMA={self._ema_val:.6g}, best={self.best_ema:.6g})")
                        if self._bad_checks > self.es_patience:
                            print(f"[EarlyStop] Stopping at epoch {epoch}: EMA plateaued.")
                            stop_training = True

                torch.cuda.empty_cache()
                gc.collect()

            if stop_training:
                break

        # Persist metrics (handles missing output_path gracefully)
        folder = os.path.dirname(output_path) if output_path else "."
        fname = os.path.basename(output_path) if output_path else ""
        parts = fname.split("_") if fname else []
        fold = f"_{parts[2]}" if len(parts) == 3 else ""

        with open(os.path.join(folder, f'val_performance{fold}.pkl'), 'wb') as handle:
            pickle.dump(self.performance, handle)

        with open(os.path.join(folder, f'epoch_loss{fold}.pkl'), 'wb') as handle:
            pickle.dump(self.loss, handle)

    def get_best_model(self, which: str = "ema"):
        """
        which: "ema" or "pearson"
        """
        if which == "pearson" and self.best_model_by_pearson is not None:
            return self.best_model_by_pearson
        if which == "ema" and self.best_model_by_ema is not None:
            return self.best_model_by_ema
        # Fallback for backward compatibility
        return self.best_model

    def evaluate(self, model, dataloader, epoch, name="Validation"):
        """
        Returns pearson_per_drug (unchanged) and stores:
          - per-drug Pearson/Spearman
          - global val_loss
          - per-drug val_loss in self.performance[epoch]["val_loss_per_drug"]
          - (set in train()) current LR stored at this epoch in self.performance[epoch]["lr"]
        """
        trues = []
        results = []
        dataloader_with_tqdm = tqdm(dataloader)

        test_df = dataloader.dataset.drug_response_df.reset_index()
        test_grouped = test_df.reset_index().groupby(1)

        model.to(self.device)
        model.eval()

        # Track global validation loss (sample-weighted mean)
        total_loss_sum = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in dataloader_with_tqdm:
                targets_cpu = (batch['response_mean'] + batch['response_residual']).to(torch.float32)
                trues.append(targets_cpu.cpu())

                batch = move_to(batch, self.device)
                preds = model(
                    batch['genotype'], batch['drug'],
                    self.nested_subtrees_forward, self.nested_subtrees_backward,
                    self.gene2system_mask, self.system2gene_mask,
                    sys2cell=self.args.sys2cell,
                    cell2sys=self.args.cell2sys,
                    sys2gene=self.args.sys2gene,
                    gene2drug=self.args.gene2drug,
                    mut2gene=self.args.mut2gene,
                    with_indices=self.args.with_indices
                )

                # batch-level validation loss (SmoothL1)
                targets = (batch['response_mean'] + batch['response_residual']).to(torch.float32).to(self.device)
                batch_loss = self.drug_response_loss(preds[:, 0], targets)
                bs = targets.numel()
                total_loss_sum += batch_loss.detach().item() * bs
                total_count += bs

                results.append(preds.detach().cpu().numpy())
                dataloader_with_tqdm.set_description(f"{name} epoch: {epoch}")

                # cleanup
                del preds, batch_loss, batch, targets, targets_cpu

        trues = torch.cat(trues).numpy()
        results = np.concatenate(results)[:, 0]

        # Global metrics (unchanged)
        r_square = metrics.r2_score(trues, results)
        pearson_global = pearsonr(trues, results)[0]
        spearman_global = spearmanr(trues, results).correlation
        print("R_square: ", r_square)
        print("Pearson R", pearson_global)
        print("Spearman Rho: ", spearman_global)

        # Per-drug metrics + per-drug validation loss
        r2_score_dict = {}
        pearson_dict = {}
        spearman_dict = {}
        val_loss_per_drug = {}

        # CPU SmoothL1Loss once (avoids device issues)
        loss_fn_cpu = nn.SmoothL1Loss(self.beta)

        test_grouped_groups = test_grouped.groups
        for smiles, indice in test_grouped_groups.items():
            # Per-drug loss (compute even if one sample)
            y_true = torch.from_numpy(trues[indice]).to(torch.float32)
            y_pred = torch.from_numpy(results[indice]).to(torch.float32)
            per_loss = loss_fn_cpu(y_pred, y_true).item()
            val_loss_per_drug[smiles] = per_loss

            # Per-drug correlations (only meaningful if >1 sample)
            if len(indice) > 1:
                r2 = metrics.r2_score(test_df.loc[indice][2], results[indice])
                rho = spearmanr(test_df.loc[indice][2], results[indice]).correlation
                p = pearsonr(test_df.loc[indice][2], results[indice])[0]
                if not np.isnan(r2):
                    r2_score_dict[smiles] = r2
                    spearman_dict[smiles] = rho
                    pearson_dict[smiles] = p

        pearson_per_drug = np.array(list(pearson_dict.values())).mean() if pearson_dict else np.nan
        spearman_per_drug = np.array(list(spearman_dict.values())).mean() if spearman_dict else np.nan
        print("Pearson per drug: ", pearson_per_drug)
        print("Spearman per drug: ", spearman_per_drug)

        # Global validation loss
        val_loss = (total_loss_sum / total_count) if total_count > 0 else float('nan')
        print("Validation SmoothL1 loss: ", val_loss)

        # Store all validation stats for this epoch (adds per-drug val loss)
        prev = self.performance.get(epoch, {})
        prev.update({
            "pearson_per_drug": pearson_dict,
            "spearman_per_drug": spearman_dict,
            "val_loss_per_drug": val_loss_per_drug,
            "val_loss": val_loss
        })
        self.performance[epoch] = prev

        # Return the selection signal: mean Pearson across drugs
        return pearson_per_drug

    def train_epoch(self, epoch):
        self.drug_response_model.train()
        self.iter_minibatches(self.drug_response_dataloader_drug, epoch, name="DrugBatch", ccc=False, feature_loss=False)

    def iter_minibatches(self, dataloader, epoch, name="", ccc=True, feature_loss=True):
        """
        NOTE: ccc and feature_loss args are kept for API compatibility but ignored.
        """
        running_loss = 0.0
        n_batches = 0

        dataloader_with_tqdm = tqdm(dataloader)
        for batch in dataloader_with_tqdm:
            batch = move_to(batch, self.device)

            preds = self.drug_response_model(
                batch['genotype'], batch['drug'],
                self.nested_subtrees_forward, self.nested_subtrees_backward,
                self.gene2system_mask, self.system2gene_mask,
                sys2cell=self.args.sys2cell,
                cell2sys=self.args.cell2sys,
                sys2gene=self.args.sys2gene,
                gene2drug=self.args.gene2drug,
                mut2gene=self.args.mut2gene,
                with_indices=self.args.with_indices
            )

            targets = (batch['response_mean'] + batch['response_residual']).to(torch.float32).to(self.device)
            loss = self.drug_response_loss(preds[:, 0], targets)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.drug_response_model.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()  # batch-wise scheduler step

            if self.fix_embedding:
                # Re-freeze embeddings after optimizer step
                self.drug_response_model.system_embedding = self.system_embedding
                self.drug_response_model.gene_embedding = self.gene_embedding

            running_loss += loss.detach().item()
            n_batches += 1

            avg_loss = running_loss / n_batches
            dataloader_with_tqdm.set_description(
                f"{name} Train epoch: {epoch}, Drug Response loss: {avg_loss:.4f}"
            )

            # cleanup
            del loss, preds, batch, targets

        # Store mean train loss for the epoch
        self.loss[epoch] = running_loss / max(n_batches, 1)
