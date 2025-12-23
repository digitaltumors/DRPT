#!/usr/bin/env python3
"""
Leiden clustering on system embeddings per drug (GPU kNN) + overrepresentation tests:
  - Global CN-burden (high/low),
  - Global Mutation-burden (high/low),
  - Predicted AUC (resistant/high vs sensitive/low),
  - Actual AUC (resistant/high vs sensitive/low).

Also saves a cluster-membership dictionary mapping:
  membership[drug][system][cluster_id] -> list of cell-line IDs

Outputs:
  1) CSV with columns:
     drug, system, cluster_id, n_cells,
     cn_direction, cn_p, cn_fdr,
     mut_direction, mut_p, mut_fdr,
     pred_direction, pred_p, pred_fdr,
     actual_direction, actual_p, actual_fdr
  2) Pickle file containing the membership dict.

Requirements:
  numpy, pandas, torch, scipy, statsmodels, igraph, leidenalg
"""

import argparse
import pickle
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from scipy.sparse import coo_matrix
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import igraph as ig
import leidenalg as la


# ---------------------------
# I/O helpers for single-column TAB files of "0,1,1,0,..."
# ---------------------------
def _read_single_column_series(path: str) -> pd.Series:
    df = pd.read_csv(path, header=None, sep="\t", engine="python", dtype=str)
    if df.shape[1] != 1:
        raise ValueError(f"{path} should have exactly one column; got {df.shape[1]}")
    return df.iloc[:, 0].astype(str).str.strip()

def _count_ones_and_len(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    ones = s.str.count('1').astype(int)
    n_tokens = (s.str.count(',') + 1).astype(int)
    return ones, n_tokens

def load_global_cnburden(
    amp_file: str,
    del_file: str,
    cell_index_file: str,
    n_genes: Optional[int] = 718,
) -> pd.Series:
    """
    Global CN-burden per cell-line = (sum amps + sum dels) / n_genes.
    Returns a Series indexed by cell-line name.
    """
    amp_s = _read_single_column_series(amp_file)
    del_s = _read_single_column_series(del_file)

    idx_df = pd.read_csv(cell_index_file, header=None, sep="\t", engine="python", dtype=str).iloc[:, :2]
    idx_df.columns = ["index", "cell_line"]
    idx_df["index"] = pd.to_numeric(idx_df["index"], errors="coerce").astype(int)

    if len(amp_s) != len(idx_df) or len(del_s) != len(idx_df):
        raise ValueError(f"Row mismatch: amp({len(amp_s)}), del({len(del_s)}), cells({len(idx_df)})")

    amp_ones, nA = _count_ones_and_len(amp_s)
    del_ones, nD = _count_ones_and_len(del_s)
    if n_genes is None:
        n_genes = int(pd.concat([nA, nD]).mode().iloc[0])

    cnburden_vals = (amp_ones + del_ones).to_numpy(dtype=float) / float(n_genes)
    cnburden = pd.Series(cnburden_vals, index=idx_df["cell_line"].values, name="Global_CN_Burden").astype(float)
    cnburden = cnburden[~cnburden.index.duplicated(keep='first')]
    return cnburden

def load_global_mutburden(
    mut_file: str,
    cell_index_file: str,
    n_genes: Optional[int] = 718,
) -> pd.Series:
    """
    Global Mutation burden per cell-line = (sum of mutations across genes) / n_genes.
    Returns a Series indexed by cell-line name.
    """
    mut_s = _read_single_column_series(mut_file)

    idx_df = pd.read_csv(cell_index_file, header=None, sep="\t", engine="python", dtype=str).iloc[:, :2]
    idx_df.columns = ["index", "cell_line"]
    idx_df["index"] = pd.to_numeric(idx_df["index"], errors="coerce").astype(int)

    if len(mut_s) != len(idx_df):
        raise ValueError(f"Row mismatch: mut({len(mut_s)}), cells({len(idx_df)})")

    mut_ones, nM = _count_ones_and_len(mut_s)
    if n_genes is None:
        n_genes = int(nM.mode().iloc[0])

    mutburden_vals = mut_ones.to_numpy(dtype=float) / float(n_genes)
    mutburden = pd.Series(mutburden_vals, index=idx_df["cell_line"].values, name="Global_Mutation_Burden").astype(float)
    mutburden = mutburden[~mutburden.index.duplicated(keep='first')]
    return mutburden


# ---------------------------
# Embeddings: per system for ONE drug
# ---------------------------
def build_system_embeddings_for_drug(results: Dict, drug: str, system_name: str, system_to_index: Dict[str, int]) -> pd.DataFrame:
    """
    results[drug]["celllines"] -> list[str]
    results[drug]["system_embeddings"] -> np.ndarray (N_cells, 131, 128)
    """
    sys_idx = system_to_index[system_name]
    cells = results[drug]["celllines"]
    emb3d = results[drug]["system_embeddings"]      # (n_cells, 131, 128)
    emb2d = emb3d[:, sys_idx, :]                    # (n_cells, 128)
    emb_df = pd.DataFrame(emb2d, index=pd.Index(cells, name="cell_line"))
    emb_df.columns = [f"e{i}" for i in range(emb_df.shape[1])]
    emb_df = emb_df[~emb_df.index.duplicated(keep='first')]
    return emb_df


# ---------------------------
# kNN graph (cosine), GPU-accelerated via PyTorch
# ---------------------------
def torch_knn_cosine_coo(emb_df: pd.DataFrame, k: int, device: Optional[str] = None) -> coo_matrix:
    """
    Build a kNN graph (cosine similarity) on GPU if available.
    Returns a COO sparse matrix with edge weights in [0,1].
    """
    N, D = emb_df.shape
    if N < 2:
        return coo_matrix((N, N))
    k_eff = min(k, N - 1)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(emb_df.values).to(device=device, dtype=torch.float32)
    X = F.normalize(X, p=2, dim=1)

    S = X @ X.t()
    S.fill_diagonal_(-1.0)

    vals, idxs = torch.topk(S, k=k_eff, dim=1, largest=True, sorted=False)
    row = torch.arange(N, device=device).unsqueeze(1).expand_as(idxs).reshape(-1)
    col = idxs.reshape(-1)
    sim = torch.clamp(vals.reshape(-1), min=0.0, max=1.0)

    r = row.detach().cpu().numpy()
    c = col.detach().cpu().numpy()
    w = sim.detach().cpu().numpy()
    W = coo_matrix((w, (r, c)), shape=(N, N))

    W = (W + W.T).tocoo()
    return W


# ---------------------------
# Leiden clustering from COO similarity
# ---------------------------
def leiden_from_coo(W: coo_matrix, resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Convert a (symmetric) COO similarity matrix to an igraph Graph and run Leiden.
    Returns an array of community labels (cluster ids) of length N.
    """
    W = W.tocoo()
    N = W.shape[0]
    edges = [(int(i), int(j)) for i, j in zip(W.row, W.col) if i != j]
    weights = [float(w) for w, i, j in zip(W.data, W.row, W.col) if i != j]

    g = ig.Graph(n=N, edges=edges, directed=False)
    g.es["weight"] = weights

    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed
    )
    labels = np.array(part.membership, dtype=int)
    return labels


# ---------------------------
# Stats: Wilcoxon signed-rank test vs population median
# ---------------------------
def signedrank_vs_median(values: np.ndarray, pop_median: float) -> float:
    """
    Two-sided Wilcoxon signed-rank test for whether median(values) differs from pop_median.
    Operates on differences (values - pop_median). Returns p-value.
    """
    v = values.astype(float) - float(pop_median)
    v = v[~np.isclose(v, 0.0)]
    if v.size == 0:
        return np.nan
    try:
        stat, p = wilcoxon(v, zero_method="wilcox", alternative="two-sided", correction=False, mode="auto")
        return float(p)
    except Exception:
        return np.nan


# ---------------------------
# Main pipeline
# ---------------------------
def run(
    results_path: str,
    system_genes_path: str,
    nest_to_name_path: str,
    index_to_nest_path: str,
    amp_file: str,
    del_file: str,
    mut_file: str,
    cell_index_file: str,
    out_csv: str,
    clusters_pickle: str,
    k: int = 10,
    resolution: float = 1.0,
    seed: int = 42,
    device: Optional[str] = None,
) -> pd.DataFrame:

    # Load results dict (single dict: results[drug] -> {...})
    with open(results_path, "rb") as fh:
        results = pickle.load(fh)

    # Load mappings, build system_to_index
    with open(nest_to_name_path, "rb") as fh:
        nest_to_name = pickle.load(fh)
    with open(index_to_nest_path, "rb") as fh:
        index_to_nest = pickle.load(fh)
    system_to_index = {nest_to_name[nest_id]: index for index, nest_id in index_to_nest.items()}

    # Load burdens once (global across the panel)
    cnb = load_global_cnburden(amp_file=amp_file, del_file=del_file, cell_index_file=cell_index_file, n_genes=718)
    mutb = load_global_mutburden(mut_file=mut_file, cell_index_file=cell_index_file, n_genes=718)

    out_rows = []
    membership: Dict[str, Dict[str, Dict[int, list]]] = {}  # drug -> system -> cluster_id -> [cell_lines]

    # Iterate drugs
    for drug in sorted(results.keys()):
        entry = results[drug]
        if not all(k in entry for k in ("celllines", "predictions", "actual", "system_embeddings")):
            continue

        # AUC series aligned to cell names
        cells = pd.Index(entry["celllines"], name="cell_line")
        pred_auc = pd.Series(np.asarray(entry["predictions"], dtype=float), index=cells, name="pred_auc")
        act_auc  = pd.Series(np.asarray(entry["actual"], dtype=float),      index=cells, name="act_auc")
        pred_auc = pred_auc[~pred_auc.index.duplicated(keep='first')]
        act_auc  = act_auc[~act_auc.index.duplicated(keep='first')]

        membership.setdefault(drug, {})

        # Iterate systems
        for system_name in sorted(system_to_index.keys()):
            emb_df = build_system_embeddings_for_drug(results, drug, system_name, system_to_index)
            if emb_df.empty:
                continue

            # Align shared cells across all signals for this (drug, system)
            common = emb_df.index
            common = common.intersection(pred_auc.index).intersection(act_auc.index).intersection(cnb.index).intersection(mutb.index)
            if len(common) < 5:
                continue

            emb_df_sub = emb_df.loc[common].copy()
            pred_sub   = pred_auc.loc[common].copy()
            act_sub    = act_auc.loc[common].copy()
            cnb_sub    = cnb.loc[common].copy()
            mut_sub    = mutb.loc[common].copy()

            # Build kNN graph (GPU if available) & Leiden clustering
            W = torch_knn_cosine_coo(emb_df_sub, k=k, device=device)
            labels = leiden_from_coo(W, resolution=resolution, seed=seed)
            clusters = pd.Series(labels, index=emb_df_sub.index, name="cluster")

            # Save cluster membership
            membership[drug].setdefault(system_name, {})
            for cid in np.unique(labels):
                idx = emb_df_sub.index[clusters.values == cid]
                membership[drug][system_name][int(cid)] = idx.tolist()

            # Population medians
            med_cnb  = float(np.median(cnb_sub.values))
            med_mut  = float(np.median(mut_sub.values))
            med_pred = float(np.median(pred_sub.values))
            med_act  = float(np.median(act_sub.values))

            # Per-cluster tests
            for cid in np.unique(labels):
                idx = emb_df_sub.index[clusters.values == cid]
                if len(idx) < 3:
                    continue

                vals_cnb  = cnb_sub.loc[idx].values
                vals_mut  = mut_sub.loc[idx].values
                vals_pred = pred_sub.loc[idx].values
                vals_act  = act_sub.loc[idx].values

                # Directions
                dir_cnb  = "high" if np.median(vals_cnb)  > med_cnb  else "low"
                dir_mut  = "high" if np.median(vals_mut)  > med_mut  else "low"
                dir_pred = "resistant" if np.median(vals_pred) > med_pred else "sensitive"
                dir_act  = "resistant" if np.median(vals_act)  > med_act  else "sensitive"

                # Two-sided Wilcoxon vs population medians
                p_cnb  = signedrank_vs_median(vals_cnb,  med_cnb)
                p_mut  = signedrank_vs_median(vals_mut,  med_mut)
                p_pred = signedrank_vs_median(vals_pred, med_pred)
                p_act  = signedrank_vs_median(vals_act,  med_act)

                out_rows.append({
                    "drug": drug,
                    "system": system_name,
                    "cluster_id": int(cid),
                    "n_cells": int(len(idx)),
                    "cn_direction": dir_cnb,  "cn_p":  p_cnb,
                    "mut_direction": dir_mut, "mut_p": p_mut,
                    "pred_direction": dir_pred, "pred_p": p_pred,
                    "actual_direction": dir_act, "actual_p": p_act,
                })

        # FDR within drug, per endpoint family
        if out_rows:
            df_drug = pd.DataFrame([r for r in out_rows if r["drug"] == drug])

            for p_col, fdr_col in [
                ("cn_p", "cn_fdr"),
                ("mut_p", "mut_fdr"),
                ("pred_p", "pred_fdr"),
                ("actual_p", "actual_fdr"),
            ]:
                mask = df_drug[p_col].notna().values
                if mask.sum() > 0:
                    _, q, _, _ = multipletests(df_drug.loc[mask, p_col], alpha=0.05, method="fdr_bh")
                    df_drug.loc[mask, fdr_col] = q
                else:
                    df_drug[fdr_col] = np.nan

            # Write back FDRs
            for _, row in df_drug.iterrows():
                for r in out_rows:
                    if (r["drug"] == row["drug"] and r["system"] == row["system"] and r["cluster_id"] == row["cluster_id"]):
                        r["cn_fdr"]     = float(row["cn_fdr"])     if pd.notna(row.get("cn_fdr", np.nan)) else np.nan
                        r["mut_fdr"]    = float(row["mut_fdr"])    if pd.notna(row.get("mut_fdr", np.nan)) else np.nan
                        r["pred_fdr"]   = float(row["pred_fdr"])   if pd.notna(row.get("pred_fdr", np.nan)) else np.nan
                        r["actual_fdr"] = float(row["actual_fdr"]) if pd.notna(row.get("actual_fdr", np.nan)) else np.nan

    # Final DataFrame & save
    cols = [
        "drug","system","cluster_id","n_cells",
        "cn_direction","cn_p","cn_fdr",
        "mut_direction","mut_p","mut_fdr",
        "pred_direction","pred_p","pred_fdr",
        "actual_direction","actual_p","actual_fdr",
    ]
    if not out_rows:
        df_out = pd.DataFrame(columns=cols)
    else:
        df_out = pd.DataFrame(out_rows)
        for c in ("cn_fdr","mut_fdr","pred_fdr","actual_fdr"):
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out = df_out[cols].sort_values(["drug","system","cluster_id"]).reset_index(drop=True)

    # Save outputs
    df_out.to_csv(out_csv, index=False)
    with open(clusters_pickle, "wb") as fh:
        pickle.dump(membership, fh)

    print(f"Wrote {len(df_out)} rows to {out_csv}")
    print(f"Saved cluster membership dict to {clusters_pickle}")
    return df_out


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Leiden clustering on system embeddings per drug + overrepresentation tests (CN-burden, Mutation-burden, AUC) with cluster membership export.")
    p.add_argument('--results',        dest='results',        type=str, default="interpretation/embeddings/results_diff_embeddings.pkl",
                   help="Pickle path to results dict (results[drug]{celllines, predictions, actual, system_embeddings})")
    p.add_argument('--system_to_genes',dest='system_genes',   type=str, default="mappings/system_to_genes.pkl",
                   help="(Not used directly here; kept for parity)")
    p.add_argument('--nest_to_name',   dest='nest_to_name',   type=str, default="mappings/nest_to_old_name.pkl",
                   help="Pickle mapping: nest_id -> system_name")
    p.add_argument('--index_to_nest',  dest='index_to_nest',  type=str, default="mappings/index_to_nest.pkl",
                   help="Pickle mapping: system_index -> nest_id")

    # CNA / Mutation inputs
    p.add_argument('--amp_file',       dest='amp_file',       type=str, default="data/cell2cnamplification_ctg_av.txt",
                   help="TAB file: each row '0,1,1,...' for amplifications")
    p.add_argument('--del_file',       dest='del_file',       type=str, default="data/cell2cndeletion_ctg_av.txt",
                   help="TAB file: each row '0,1,1,...' for deletions")
    p.add_argument('--mut_file',       dest='mut_file',       type=str, default="data/cell2mutation_ctg_av.txt",
                   help="TAB file: each row '0,1,1,...' for mutations")
    p.add_argument('--cell_index',     dest='cell_index',     type=str, default="data/cell2ind_av.txt",
                   help="TAB 2-col: [index, cell_line_name]")

    # Graph/Leiden params
    p.add_argument('--k',              dest='k',              type=int,   default=15,    help="k for kNN graph")
    p.add_argument('--resolution',     dest='resolution',     type=float, default=0.8,   help="Leiden resolution parameter")
    p.add_argument('--seed',           dest='seed',           type=int,   default=42,    help="Random seed for Leiden")

    # Device & outputs
    p.add_argument('--device',         dest='device',         type=str,   default=None,  help="'cuda' or 'cpu' (auto if omitted)")
    p.add_argument('--out',            dest='out',            type=str,   default="cluster_overrep_by_drug.csv",
                   help="Output CSV path")
    p.add_argument('--clusters_pickle', dest='clusters_pickle', type=str, default="cluster_membership_by_drug.pkl",
                   help="Output pickle path for cluster membership dictionary")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        results_path=args.results,
        system_genes_path=args.system_genes,
        nest_to_name_path=args.nest_to_name,
        index_to_nest_path=args.index_to_nest,
        amp_file=args.amp_file,
        del_file=args.del_file,
        mut_file=args.mut_file,
        cell_index_file=args.cell_index,
        out_csv=args.out,
        clusters_pickle=args.clusters_pickle,
        k=args.k,
        resolution=args.resolution,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()
