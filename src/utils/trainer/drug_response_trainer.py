import os
import gc
import copy
import pickle
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

        # Optimizer / scheduler
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.drug_response_model.parameters()),
            lr=args.lr, weight_decay=args.wd
        )
        self.scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer, base_lr=args.lr / 10, max_lr=args.lr, cycle_momentum=False
        )

        # Data / masks
        self.validation_dataloader = validation_dataloader
        self.args = args
        self.l2_lambda = args.l2_lambda
        self.total_train_step = len(self.drug_response_dataloader_drug) * args.epochs

        tp = self.drug_response_dataloader_drug.dataset.tree_parser
        self.nested_subtrees_forward = move_to(tp.get_nested_subtree_mask(args.subtree_order, direction='forward'), device)
        self.nested_subtrees_backward = move_to(tp.get_nested_subtree_mask(args.subtree_order, direction='backward'), device)
        self.gene2system_mask = move_to(torch.tensor(tp.gene2sys_mask, dtype=torch.bool), device)
        self.system2gene_mask = move_to(torch.tensor(tp.sys2gene_mask, dtype=torch.bool), device)
        print("%d sys2gene in Dataloader" % tp.sys2gene_mask.sum())

        # Book-keeping
        self.best_model = self.drug_response_model
        self.fix_embedding = fix_embedding
        self.g2p_module_names = ["Mut2Sys", "Sys2Cell", "Cell2Sys"]  # preserved for compatibility
        self.performance = {}  # {epoch: {"pearson_per_drug": {...}, "spearman_per_drug": {...}, "val_loss": float, "val_loss_per_drug": {...}}}
        self.loss = {}         # {epoch: mean_train_loss}

        # Optional: keep embeddings fixed during training
        if fix_embedding:
            if self.args.multiprocessing_distributed:
                self.system_embedding = copy.deepcopy(self.drug_response_model.module.system_embedding)
                self.gene_embedding = copy.deepcopy(self.drug_response_model.module.gene_embedding)
            else:
                self.system_embedding = copy.deepcopy(self.drug_response_model.system_embedding)
                self.gene_embedding = copy.deepcopy(self.drug_response_model.gene_embedding)

    def train(self, epochs, output_path=None):

        self.best_model = self.drug_response_model
        best_performance = 0.0

        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)
            gc.collect()
            torch.cuda.empty_cache()

            # Validate on schedule
            if (epoch % self.args.val_step) == 0 and (self.validation_dataloader is not None):
                performance = self.evaluate(self.drug_response_model, self.validation_dataloader, epoch, name="Validation")
                if performance > best_performance:
                    # keep a CPU copy for checkpointing/saving outside GPU context
                    self.best_model = copy.deepcopy(self.drug_response_model).to('cpu')
                    best_performance = performance
                torch.cuda.empty_cache()
                gc.collect()

            # Save model checkpoints on schedule
            if (epoch % self.args.val_step) == 0:
                if (not self.args.multiprocessing_distributed) or (
                    self.args.multiprocessing_distributed and self.args.rank % torch.cuda.device_count() == 0
                ):
                    if output_path:
                        output_path_epoch = f"{output_path}.{epoch}"
                        print("Save to...", output_path_epoch)
                        state = {"arguments": self.args}
                        if self.args.multiprocessing_distributed:
                            state["state_dict"] = self.drug_response_model.module.state_dict()
                        else:
                            state["state_dict"] = self.drug_response_model.state_dict()
                        torch.save(state, output_path_epoch)

        # Persist metrics (handles missing output_path gracefully)
        folder = os.path.dirname(output_path) if output_path else "."
        fname = os.path.basename(output_path) if output_path else ""
        parts = fname.split("_") if fname else []
        fold = f"_{parts[2]}" if len(parts) == 3 else ""

        with open(os.path.join(folder, f'val_performance{fold}.pkl'), 'wb') as handle:
            pickle.dump(self.performance, handle)

        with open(os.path.join(folder, f'epoch_loss{fold}.pkl'), 'wb') as handle:
            pickle.dump(self.loss, handle)

    def get_best_model(self):
        return self.best_model

    def evaluate(self, model, dataloader, epoch, name="Validation"):
        """
        Returns pearson_per_drug (unchanged) and stores:
          - per-drug Pearson/Spearman
          - global val_loss
          - per-drug val_loss in self.performance[epoch]["val_loss_per_drug"]
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

        # Per-drug metrics + NEW: per-drug validation loss
        r2_score_dict = {}
        pearson_dict = {}
        spearman_dict = {}
        val_loss_per_drug = {}

        # CPU SmoothL1Loss once (avoids device issues)
        loss_fn_cpu = nn.SmoothL1Loss(self.beta)

        for smiles, indice in test_grouped.groups.items():
            # Per-drug loss (compute even if only one sample)
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
        self.performance[epoch] = {
            "pearson_per_drug": pearson_dict,
            "spearman_per_drug": spearman_dict,
            "val_loss_per_drug": val_loss_per_drug,
            "val_loss": val_loss
        }

        # Maintain original interface
        return pearson_per_drug

    def train_epoch(self, epoch):
        self.drug_response_model.train()
        # Keep signature usage the same
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
            self.scheduler.step()

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