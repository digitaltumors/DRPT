#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from typing import Dict, Tuple, List
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

# ---------------------------
# Helpers
# ---------------------------

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR for a 1D array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    q_sorted = (p_sorted * n) / (np.arange(1, n + 1))
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return np.minimum(q, 1.0)

def pearson_cols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation between each column of X (n x S) and y (n,)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    num = np.sum(Xc * yc[:, None], axis=0)
    denom = np.sqrt(np.sum(Xc**2, axis=0) * np.sum(yc**2))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / denom
    r = np.where(np.isfinite(r), r, 0.0)
    return r

def rank_array(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average").values

def rank_columns(M: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(rank_array, 0, M)

def load_system_mappings(nest_to_name_path: str, index_to_nest_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(nest_to_name_path, "rb") as h:
        nest_to_name = pickle.load(h)
    with open(index_to_nest_path, "rb") as h:
        index_to_nest = pickle.load(h)
    system_to_index = {nest_to_name[nest_id]: idx for idx, nest_id in index_to_nest.items()}
    index_to_system = {idx: nest_to_name[nest_id] for idx, nest_id in index_to_nest.items()}
    return system_to_index, index_to_system

def load_gene_index(gene_index_file: str) -> Dict[int, str]:
    """Two-column TSV: index<TAB>gene_name -> {index: gene}"""
    df = pd.read_csv(gene_index_file, sep="\t", header=None, names=["index", "gene"], dtype={0:int,1:str})
    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns in {gene_index_file}.")
    return dict(zip(df["index"].astype(int), df["gene"].astype(str)))

def load_cell_index(cell2ind_path: str) -> Dict[str, int]:
    """Reads two-column txt (index, cell_line) -> {cell_line: index}."""
    cell2ind = {}
    with open(cell2ind_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                idx_str, name = line.split(",", 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                idx_str, name = parts[0], parts[1]
            cell2ind[name] = int(idx_str)
    return cell2ind

def load_alteration_strings(path: str) -> List[str]:
    """Each line is a comma-separated 0/1 string: '0,1,0,...'."""
    lines = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines

def burden_from_strings(entry: str) -> int:
    return sum(1 for tok in entry.split(",") if tok.strip() == "1")

# ---------------------------
# Burden assembly
# ---------------------------

def compute_burdens_for_drug(drug_celllines: List[str],
                             cell2ind: Dict[str, int],
                             mut_lines: List[str],
                             amp_lines: List[str],
                             del_lines: List[str],
                             total_genes: int = 718) -> Tuple[np.ndarray, np.ndarray]:
    n = len(drug_celllines)
    C = np.zeros(n, dtype=float)
    M = np.zeros(n, dtype=float)
    for i, cl in enumerate(drug_celllines):
        idx = cell2ind[cl]
        mut = burden_from_strings(mut_lines[idx])
        amp = burden_from_strings(amp_lines[idx])
        dele = burden_from_strings(del_lines[idx])
        C[i] = (amp + dele) / float(total_genes)  # CN-burden
        M[i] = (mut) / float(total_genes)         # mutation burden
    return C, M

def residualize_Y_on_covariates(y: np.ndarray, C: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Semi-partial setup: residualize Y only on [1, C, M]."""
    X = np.c_[np.ones_like(C), C, M]
    lr = LinearRegression()
    lr.fit(X, y)
    return y - lr.predict(X)

# ---------------------------
# Semi-partial correlations (Y residualized ONLY)
# ---------------------------

def compute_semi_partial_for_drug(attn: np.ndarray,
                                  preds: np.ndarray,
                                  C: np.ndarray,
                                  M: np.ndarray,
                                  rng: np.random.Generator,
                                  iterations: int = 50000):
    """
    Semi-partial importance:
      - residualize Y_pred on CN & MUT: Y_res = res(Y | C,M)
      - Pearson: corr(attn_j, Y_res) for each feature j
      - Spearman: corr(rank(attn_j), rank(Y_res)) for each feature j
      - Null (per drug): permute indices of Y_res; pool mean/std across features & permutations
    Returns:
      r_sp_pearson (S,), r_sp_spearman (S,),
      null_stats_pearson {'mean','std'}, null_stats_spearman {'mean','std'}
    """
    n, S = attn.shape

    # Residualize Y only
    Y_res = residualize_Y_on_covariates(preds, C, M)

    # Pearson semi-partial: raw attn vs Y_res
    r_sp_pearson = pearson_cols(attn, Y_res)

    # Spearman semi-partial: rank-transform both then Pearson
    A_r = rank_columns(attn)
    Y_res_r = rank_array(Y_res)
    r_sp_spearman = pearson_cols(A_r, Y_res_r)

    # Permutation nulls by permuting Y_res
    total = iterations * S
    pooled_sum_p = 0.0
    pooled_sumsq_p = 0.0
    pooled_sum_s = 0.0
    pooled_sumsq_s = 0.0
    idx = np.arange(n)

    for _ in range(iterations):
        rng.shuffle(idx)
        r_perm_p = pearson_cols(attn, Y_res[idx])
        pooled_sum_p   += float(r_perm_p.sum())
        pooled_sumsq_p += float((r_perm_p**2).sum())

        r_perm_s = pearson_cols(A_r, Y_res_r[idx])
        pooled_sum_s   += float(r_perm_s.sum())
        pooled_sumsq_s += float((r_perm_s**2).sum())

    mu_p = pooled_sum_p / total
    var_p = max(pooled_sumsq_p / total - mu_p**2, 0.0)
    sd_p  = sqrt(var_p)

    mu_s = pooled_sum_s / total
    var_s = max(pooled_sumsq_s / total - mu_s**2, 0.0)
    sd_s  = sqrt(var_s)

    return r_sp_pearson, r_sp_spearman, {"mean": mu_p, "std": sd_p}, {"mean": mu_s, "std": sd_s}

# ---------------------------
# Pipeline (systems or genes)
# ---------------------------

def run_pipeline(results_path: str,
                 score: str,
                 nest_to_name_path: str,
                 index_to_nest_path: str,
                 gene_index_file: str,
                 cell2ind_path: str,
                 mut_path: str,
                 del_path: str,
                 amp_path: str,
                 iterations: int = 50000,
                 seed: int = 42,
                 out_csv: str = "semi_partial_importance.csv") -> pd.DataFrame:

    with open(results_path, "rb") as h:
        results = pickle.load(h)

    drugs = list(results.keys())
    if not drugs:
        raise ValueError("No drugs found in results.")

    # Choose feature space
    if score == "systems":
        _, index_to_system = load_system_mappings(nest_to_name_path, index_to_nest_path)
        S = results[drugs[0]]["system_attn"].shape[1]
        feature_names = [index_to_system[i] for i in range(S)]
        feat_key = "system_attn"
        feature_label = "system"
    elif score == "genes":
        index_to_gene = load_gene_index(gene_index_file)
        S = results[drugs[0]]["gene_attn"].shape[1]
        feature_names = [index_to_gene.get(i, f"GENE_{i}") for i in range(S)]
        feat_key = "gene_attn"
        feature_label = "gene"
    else:
        raise ValueError("score must be one of: 'systems', 'genes'")

    # Burden resources
    cell2ind  = load_cell_index(cell2ind_path)
    mut_lines = load_alteration_strings(mut_path)
    del_lines = load_alteration_strings(del_path)
    amp_lines = load_alteration_strings(amp_path)

    rng = np.random.default_rng(seed)
    rows = []
    all_pvals_p = []  # for global BH (pearson)
    all_pvals_s = []  # for global BH (spearman)

    for drug in drugs:
        celllines = list(results[drug]["celllines"])
        preds = np.asarray(results[drug]["predictions"], dtype=float)
        attn  = np.asarray(results[drug][feat_key], dtype=float)

        if attn.shape[1] != S:
            raise ValueError(f"Feature count mismatch for {drug}: expected {S}, got {attn.shape[1]}")

        # Burdens for this drug's cell lines
        C, M = compute_burdens_for_drug(celllines, cell2ind, mut_lines, amp_lines, del_lines, total_genes=718)

        # Semi-partial correlations + null summaries
        r_p, r_s, null_p, null_s = compute_semi_partial_for_drug(attn, preds, C, M, rng, iterations)

        # Z-test p-values vs pooled null (two-sided)
        def ztest(obs, mu, sd):
            if sd <= 0 or not np.isfinite(sd):
                return np.ones_like(obs, dtype=float)
            z = (obs - mu) / sd
            return 2.0 * (1.0 - norm.cdf(np.abs(z)))

        p_p = ztest(r_p, null_p["mean"], null_p["std"])
        p_s = ztest(r_s, null_s["mean"], null_s["std"])

        # Per-drug MTC
        p_bonf_p = np.minimum(p_p * S, 1.0)
        p_bonf_s = np.minimum(p_s * S, 1.0)
        q_bh_p   = bh_fdr(p_p)
        q_bh_s   = bh_fdr(p_s)

        all_pvals_p.extend(p_p.tolist())
        all_pvals_s.extend(p_s.tolist())

        for j in range(S):
            rows.append({
                "drug": drug,
                feature_label: feature_names[j],
                "semi_partial_pearson":  float(r_p[j]),
                "semi_partial_spearman": float(r_s[j]),
                "p_pearson":  float(p_p[j]),
                "p_spearman": float(p_s[j]),
                "p_bonferroni_pearson":  float(p_bonf_p[j]),
                "p_bonferroni_spearman": float(p_bonf_s[j]),
                "q_fdr_pearson":  float(q_bh_p[j]),
                "q_fdr_spearman": float(q_bh_s[j]),
            })

    # Global BH across all drugs×features (separately for Pearson & Spearman)
    df = pd.DataFrame(rows)
    df["q_fdr_pearson_all"]  = bh_fdr(np.asarray(all_pvals_p, dtype=float))
    df["q_fdr_spearman_all"] = bh_fdr(np.asarray(all_pvals_s, dtype=float))

    # Order columns nicely
    out_cols = ["drug", feature_label,
                "semi_partial_pearson", "semi_partial_spearman",
                "p_pearson", "p_spearman",
                "p_bonferroni_pearson", "p_bonferroni_spearman",
                "q_fdr_pearson", "q_fdr_spearman",
                "q_fdr_pearson_all", "q_fdr_spearman_all"]
    df = df[out_cols]

    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with shape {df.shape}. Mode: {score}")
    return df

# ---------------------------
# CLI
# ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Semi-partial correlation importance (residualize Y_pred on CN & MUT only; attentions left raw). "
                    "Supports systems or genes. Z-test vs permutation null; Bonferroni & BH per drug + global BH."
    )
    ap.add_argument("--results", type=str, default="results_g2d.pkl")
    ap.add_argument("--score", type=str, choices=["systems", "genes"], default="systems",
                    help="Which feature space to score.")
    # system mappings (for --score systems)
    ap.add_argument("--nest_to_name", type=str, default="mappings/nest_to_old_name.pkl")
    ap.add_argument("--index_to_nest", type=str, default="mappings/index_to_nest.pkl")
    # gene mapping (for --score genes)
    ap.add_argument("--gene_index_file", type=str, default="data/gene2ind_ctg_av.txt",
                    help="2-column TSV: index<TAB>gene_name corresponding to gene_attn columns.")
    # burdens
    ap.add_argument("--cell2ind", type=str, default="data/cell2ind_av.txt")
    ap.add_argument("--mut", type=str, default="data/cell2mutation_ctg_av.txt")
    ap.add_argument("--del_", type=str, default="data/cell2cndeletion_ctg_av.txt")
    ap.add_argument("--amp", type=str, default="data/cell2cnamplification_ctg_av.txt")
    # null + runtime
    ap.add_argument("--iterations", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="semi_partial_importance_g2d.csv")
    args = ap.parse_args()

    run_pipeline(
        results_path=args.results,
        score=args.score,
        nest_to_name_path=args.nest_to_name,
        index_to_nest_path=args.index_to_nest,
        gene_index_file=args.gene_index_file,
        cell2ind_path=args.cell2ind,
        mut_path=args.mut,
        del_path=args.del_,
        amp_path=args.amp,
        iterations=args.iterations,
        seed=args.seed,
        out_csv=args.out_csv
    )
