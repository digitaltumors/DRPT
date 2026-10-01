#!/usr/bin/env python3

import argparse
import itertools
import math
import pickle

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------
def read_single_column_series(path: str) -> pd.Series:
    df = pd.read_csv(path, header=None, sep="\t", engine="python", dtype=str)
    if df.shape[1] != 1:
        raise ValueError(f"{path} should have exactly one column; got {df.shape[1]}")
    return df.iloc[:, 0].astype(str).str.strip()


def parse_binary_matrix(series_of_rows: pd.Series, n_cols_expected: int) -> np.ndarray:
    rows = [np.fromstring(row, sep=",", dtype=np.int8) for row in series_of_rows.values]
    arr = np.vstack(rows)

    if arr.shape[1] != n_cols_expected:
        raise ValueError(f"Parsed matrix has {arr.shape[1]} columns, expected {n_cols_expected}.")

    return arr


def load_indices(cell_index_file: str, gene_index_file: str):
    cell_df = pd.read_csv(cell_index_file, sep="\t", header=None, names=["Index", "CellLine"])
    gene_df = pd.read_csv(gene_index_file, sep="\t", header=None, names=["Index", "Gene"])

    cell_df["Index"] = pd.to_numeric(cell_df["Index"], errors="coerce").astype(int)
    gene_df["Index"] = pd.to_numeric(gene_df["Index"], errors="coerce").astype(int)

    cell_df = cell_df.sort_values("Index")
    gene_df = gene_df.sort_values("Index")

    cell_lines = pd.Index(cell_df["CellLine"].astype(str), name="cell_line")
    genes = pd.Index(gene_df["Gene"].astype(str), name="gene")

    return cell_lines, genes


def load_binary_df(path: str, cell_lines: pd.Index, genes: pd.Index) -> pd.DataFrame:
    s = read_single_column_series(path)
    arr = parse_binary_matrix(s, n_cols_expected=len(genes))

    if arr.shape[0] != len(cell_lines):
        raise ValueError(f"{path}: parsed {arr.shape[0]} rows but expected {len(cell_lines)} cell lines.")

    return pd.DataFrame(arr, index=cell_lines, columns=genes)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# FDR and p-values, SciPy-free
# ------------------------------------------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan, dtype=float)

    valid = np.isfinite(pvals)
    if valid.sum() == 0:
        return qvals

    p = pvals[valid]
    n = len(p)

    order = np.argsort(p)
    ranked_p = p[order]

    ranked_q = ranked_p * n / np.arange(1, n + 1)
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    ranked_q = np.clip(ranked_q, 0, 1)

    q = np.empty(n, dtype=float)
    q[order] = ranked_q

    qvals[valid] = q
    return qvals


def harmonic_mean_pvalue(pvals: np.ndarray) -> float:
    pvals = np.asarray(pvals, dtype=float)
    valid = np.isfinite(pvals) & (pvals > 0)
    if valid.sum() == 0:
        return np.nan

    p = np.clip(pvals[valid], np.nextafter(0, 1), 1.0)
    return float(len(p) / np.sum(1.0 / p))


def split_alteration_feature(feature_name):
    if not isinstance(feature_name, str) or "_" not in feature_name:
        return feature_name, np.nan
    gene, alteration_type = feature_name.rsplit("_", 1)
    return gene, alteration_type


def add_hmp_fdr_by_drug(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["epistasis_score_hmp_fdr"] = np.nan
    summary_df["epistasis_score_hmp_qvalue"] = np.nan
    summary_df["epistasis_score_hmp_neglog10q"] = np.nan
    summary_df["epistasis_score_hmp_fdr_n_systems"] = 0

    if summary_df.empty or "epistasis_score_hmp_p" not in summary_df.columns:
        return summary_df

    for drug, idx in summary_df.groupby("drug", sort=False).groups.items():
        hmp_pvals = summary_df.loc[idx, "epistasis_score_hmp_p"].values
        qvals = bh_fdr(hmp_pvals)
        n_valid = int(np.isfinite(hmp_pvals).sum())
        summary_df.loc[idx, "epistasis_score_hmp_fdr"] = qvals
        summary_df.loc[idx, "epistasis_score_hmp_qvalue"] = qvals
        summary_df.loc[idx, "epistasis_score_hmp_fdr_n_systems"] = n_valid

    qvals = summary_df["epistasis_score_hmp_qvalue"].clip(lower=np.nextafter(0, 1))
    valid = np.isfinite(qvals)
    summary_df.loc[valid, "epistasis_score_hmp_neglog10q"] = -np.log10(qvals[valid])
    return summary_df


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-14) -> float:
    """Continued fraction for incomplete beta from Numerical Recipes."""
    fpmin = np.finfo(float).tiny / eps
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if not (a > 0 and b > 0):
        return np.nan
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_pvalue(t_stat: float, df: int) -> float:
    """Exact two-sided Student's t-test p-value without SciPy."""
    if df <= 0 or not np.isfinite(t_stat):
        return np.nan
    x = df / (df + t_stat * t_stat)
    p = regularized_incomplete_beta(0.5 * df, 0.5, x)
    return float(np.clip(p, 0.0, 1.0))


# ------------------------------------------------------------
# Metadata lookup
# ------------------------------------------------------------
def get_drug_response(results, drug: str, response_key: str) -> pd.Series:
    if drug not in results:
        raise KeyError(f"Drug '{drug}' not found in results.")
    if response_key not in results[drug]:
        raise KeyError(f"results['{drug}'] does not contain '{response_key}'.")

    cells = pd.Index(results[drug]["celllines"], name="cell_line")
    vals = np.asarray(results[drug][response_key], dtype=float)

    y = pd.Series(vals, index=cells, name="y")
    y = y[~y.index.duplicated(keep="first")]
    y = y[pd.notna(y)]

    return y


def lookup_importance(importance_df: pd.DataFrame, drug: str, system: str):
    sub = importance_df.copy()

    if "drug" in sub.columns:
        sub = sub[sub["drug"].astype(str) == str(drug)]

    sub = sub[sub["system"].astype(str) == str(system)]

    if sub.empty:
        return np.nan, np.nan

    row = sub.iloc[0]
    importance_score = row["semi_partial_pearson"] if "semi_partial_pearson" in row.index else np.nan
    importance_p = row["p_bonferroni_pearson"] if "p_bonferroni_pearson" in row.index else np.nan

    return importance_score, importance_p


def lookup_crispr_fdr(crispr_df: pd.DataFrame, drug: str, system: str):
    sub = crispr_df.copy()

    if "drug" in sub.columns:
        sub = sub[sub["drug"].astype(str) == str(drug)]

    sub = sub[sub["system"].astype(str) == str(system)]

    if sub.empty:
        return np.nan

    return sub.iloc[0]["best_fdr"] if "best_fdr" in sub.columns else np.nan


# ------------------------------------------------------------
# Fast OLS, SciPy-free
# ------------------------------------------------------------
def fit_ols_rss(X: np.ndarray, y: np.ndarray):
    beta, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    if residuals.size == 0:
        resid = y - X @ beta
        rss = float(np.sum(resid ** 2))
    else:
        rss = float(residuals[0])

    return rss, beta, rank


def interaction_t_test_numpy(y: np.ndarray, X_full: np.ndarray):
    """
    Full model column order must be:
        intercept, x1, x2, x1:x2, covariates...

    Tests beta_interaction = 0 using an exact two-sided Student's t
    p-value computed with a SciPy-free incomplete beta implementation.
    """
    n, p = X_full.shape

    beta, residuals, rank, _ = np.linalg.lstsq(X_full, y, rcond=None)
    resid = y - X_full @ beta
    rss = float(np.sum(resid ** 2))
    df = n - rank

    if df <= 0 or rss <= 0:
        return np.nan, np.nan, beta, rss, rank

    sigma2 = rss / df
    xtx_inv = np.linalg.pinv(X_full.T @ X_full)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)

    if len(se) <= 3 or se[3] <= 0 or not np.isfinite(se[3]):
        return np.nan, np.nan, beta, rss, rank

    t_stat = beta[3] / se[3]
    pval = student_t_two_sided_pvalue(t_stat, df)

    return t_stat, pval, beta, rss, rank


def nested_interaction_test(y: np.ndarray, X_additive: np.ndarray, X_full: np.ndarray):
    """
    SciPy-free interaction test.

    Since the interaction model adds one term, the interaction t-test and
    nested F-test are equivalent up to F = t^2.
    """
    rss0, _, _ = fit_ols_rss(X_additive, y)
    t_stat, p, beta_full, rss1, _ = interaction_t_test_numpy(y, X_full)

    if not np.isfinite(t_stat):
        return np.nan, np.nan, np.nan, np.nan

    f_stat = float(t_stat ** 2)

    tss = float(np.sum((y - y.mean()) ** 2))
    if tss <= 0:
        delta_r2 = np.nan
    else:
        r2_0 = 1.0 - rss0 / tss
        r2_1 = 1.0 - rss1 / tss
        delta_r2 = r2_1 - r2_0

    interaction_beta = beta_full[3] if len(beta_full) > 3 else np.nan

    return f_stat, p, delta_r2, interaction_beta


# ------------------------------------------------------------
# Pair test
# ------------------------------------------------------------
def test_pair_fast(
    i1,
    i2,
    feature_names,
    feature_gene,
    X_values,
    y_values,
    covariate_matrix,
    include_same_gene_pairs,
    min_pair_count,
    min_cells_per_combo,
):
    f1 = feature_names[i1]
    f2 = feature_names[i2]

    gene1 = feature_gene[i1]
    gene2 = feature_gene[i2]

    x1 = X_values[:, i1].astype(float)
    x2 = X_values[:, i2].astype(float)

    n11 = int(((x1 == 1) & (x2 == 1)).sum())
    n10 = int(((x1 == 1) & (x2 == 0)).sum())
    n01 = int(((x1 == 0) & (x2 == 1)).sum())
    n00 = int(((x1 == 0) & (x2 == 0)).sum())

    if n11 < min_pair_count:
        return {
            "pair": f"{f1} x {f2}",
            "alteration_1": f1,
            "alteration_2": f2,
            "n_00": n00,
            "n_10": n10,
            "n_01": n01,
            "n_11": n11,
            "interaction_beta": np.nan,
            "f_statistic": np.nan,
            "epistasis_p": np.nan,
            "delta_r2": np.nan,
            "model_status": "skipped_low_pair_count",
        }

    if min(n00, n10, n01, n11) < min_cells_per_combo:
        return {
            "pair": f"{f1} x {f2}",
            "alteration_1": f1,
            "alteration_2": f2,
            "n_00": n00,
            "n_10": n10,
            "n_01": n01,
            "n_11": n11,
            "interaction_beta": np.nan,
            "f_statistic": np.nan,
            "epistasis_p": np.nan,
            "delta_r2": np.nan,
            "model_status": "skipped_sparse_combo",
        }

    interaction = x1 * x2
    intercept = np.ones(len(y_values))

    X0 = np.column_stack([intercept, x1, x2, covariate_matrix])
    X1 = np.column_stack([intercept, x1, x2, interaction, covariate_matrix])

    try:
        f_stat, pval, delta_r2, beta_interaction = nested_interaction_test(y_values, X0, X1)
        status = "fit" if np.isfinite(pval) else "failed_nonfinite"
    except Exception as e:
        f_stat, pval, delta_r2, beta_interaction = np.nan, np.nan, np.nan, np.nan
        status = f"failed: {str(e)}"

    return {
        "pair": f"{f1} x {f2}",
        "alteration_1": f1,
        "alteration_2": f2,
        "n_00": n00,
        "n_10": n10,
        "n_01": n01,
        "n_11": n11,
        "interaction_beta": beta_interaction,
        "f_statistic": f_stat,
        "epistasis_p": pval,
        "delta_r2": delta_r2,
        "model_status": status,
    }


# ------------------------------------------------------------
# System scan
# ------------------------------------------------------------
def compute_system_epistasis_scores_fast(
    y: pd.Series,
    system: str,
    genes_in_system,
    amp_df: pd.DataFrame,
    del_df: pd.DataFrame,
    mut_df: pd.DataFrame,
    alteration_types=("mut", "amp", "del"),
    include_cnb=True,
    include_tmb=False,
    include_same_gene_pairs=False,
    min_alt_count=5,
    min_pair_count=3,
    min_cells_per_combo=2,
    fdr_threshold=0.05,
    n_jobs=1,
):
    genes = [g for g in genes_in_system if g in amp_df.columns]

    common = y.index.intersection(amp_df.index).intersection(del_df.index).intersection(mut_df.index)
    if len(common) < 10:
        return None, pd.DataFrame()

    y_values = y.loc[common].values.astype(float)

    amp_sub = amp_df.loc[common]
    del_sub = del_df.loc[common]
    mut_sub = mut_df.loc[common]

    n_panel_genes = amp_df.shape[1]

    covariates = []

    if include_cnb:
        cnb = ((amp_sub.sum(axis=1).values + del_sub.sum(axis=1).values) / n_panel_genes).astype(float)
        covariates.append(cnb)

    if include_tmb:
        tmb = (mut_sub.sum(axis=1).values / n_panel_genes).astype(float)
        covariates.append(tmb)

    if len(covariates) > 0:
        covariate_matrix = np.column_stack(covariates)
    else:
        covariate_matrix = np.empty((len(y_values), 0))

    feature_series = {}
    feature_gene = {}

    for gene in genes:
        if "mut" in alteration_types:
            name = f"{gene}_mut"
            feature_series[name] = mut_sub[gene].astype(np.int8).values
            feature_gene[name] = gene

        if "amp" in alteration_types:
            name = f"{gene}_amp"
            feature_series[name] = amp_sub[gene].astype(np.int8).values
            feature_gene[name] = gene

        if "del" in alteration_types:
            name = f"{gene}_del"
            feature_series[name] = del_sub[gene].astype(np.int8).values
            feature_gene[name] = gene

    if len(feature_series) < 2:
        return None, pd.DataFrame()

    X_df = pd.DataFrame(feature_series, index=common)

    alt_counts = X_df.sum(axis=0)
    keep = alt_counts[alt_counts >= min_alt_count].index.tolist()
    X_df = X_df[keep]

    if X_df.shape[1] < 2:
        return None, pd.DataFrame()

    feature_names = list(X_df.columns)
    feature_gene_list = [feature_gene[f] for f in feature_names]
    X_values = X_df.values.astype(np.int8)

    pair_indices = [
        (i1, i2)
        for i1, i2 in itertools.combinations(range(len(feature_names)), 2)
        if include_same_gene_pairs or feature_gene_list[i1] != feature_gene_list[i2]
    ]

    n_pairs_candidate = len(pair_indices)
    if n_pairs_candidate == 0:
        return None, pd.DataFrame()

    if Parallel is None:
        if n_jobs not in (None, 1):
            raise ImportError("joblib is required for n_jobs != 1. Install joblib or rerun with --n_jobs 1.")
        pair_results = [
            test_pair_fast(
                i1=i1,
                i2=i2,
                feature_names=feature_names,
                feature_gene=feature_gene_list,
                X_values=X_values,
                y_values=y_values,
                covariate_matrix=covariate_matrix,
                include_same_gene_pairs=include_same_gene_pairs,
                min_pair_count=min_pair_count,
                min_cells_per_combo=min_cells_per_combo,
            )
            for i1, i2 in pair_indices
        ]
    else:
        pair_results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(test_pair_fast)(
                i1=i1,
                i2=i2,
                feature_names=feature_names,
                feature_gene=feature_gene_list,
                X_values=X_values,
                y_values=y_values,
                covariate_matrix=covariate_matrix,
                include_same_gene_pairs=include_same_gene_pairs,
                min_pair_count=min_pair_count,
                min_cells_per_combo=min_cells_per_combo,
            )
            for i1, i2 in pair_indices
        )

    pair_df = pd.DataFrame(pair_results)

    if pair_df.empty:
        return None, pair_df

    pair_df.insert(0, "system", system)
    pair_df["epistasis_fdr"] = bh_fdr(pair_df["epistasis_p"].values)

    fit_df = pair_df[pair_df["model_status"].eq("fit") & pair_df["epistasis_p"].notna()].copy()

    if fit_df.empty:
        summary = {
            "n_system_genes": len(genes),
            "n_features_tested": int(X_df.shape[1]),
            "n_pairs_candidate": n_pairs_candidate,
            "n_pairs_tested": 0,
            "n_pairs_skipped_low_pair_count": int(pair_df["model_status"].eq("skipped_low_pair_count").sum()),
            "n_pairs_skipped_sparse_combo": int(pair_df["model_status"].eq("skipped_sparse_combo").sum()),
            "n_pairs_failed": int(pair_df["model_status"].str.startswith("failed", na=False).sum()),
            "fdr_threshold": fdr_threshold,
            "n_epistasis_pairs_fdr": 0,
            "n_epistasis_pairs_fdr05": 0,
            "epistasis_score_fraction_fdr": 0.0,
            "epistasis_score_fraction_fdr05": 0.0,
            "epistasis_score_fraction_tested_fdr": np.nan,
            "epistasis_score_mean_neglog10p": np.nan,
            "epistasis_score_hmp_p": np.nan,
            "epistasis_score_hmp_neglog10p": np.nan,
            "epistasis_score_max_neglog10p": np.nan,
            "max_neglog10p_pair": np.nan,
            "max_neglog10p_gene_1": np.nan,
            "max_neglog10p_alteration_type_1": np.nan,
            "max_neglog10p_gene_2": np.nan,
            "max_neglog10p_alteration_type_2": np.nan,
            "max_neglog10p_alteration_1": np.nan,
            "max_neglog10p_alteration_2": np.nan,
            "max_neglog10p_epistasis_p": np.nan,
            "max_neglog10p_epistasis_fdr": np.nan,
            "median_abs_interaction_beta": np.nan,
            "mean_delta_r2": np.nan,
        }
        return summary, pair_df

    pvals = fit_df["epistasis_p"].clip(lower=np.nextafter(0, 1)).values
    fdrs = fit_df["epistasis_fdr"].values

    n_pairs_tested = int(fit_df.shape[0])
    n_sig = int(np.sum(fdrs < fdr_threshold))
    hmp = harmonic_mean_pvalue(pvals)
    max_pair_row = fit_df.loc[fit_df["epistasis_p"].idxmin()]
    max_gene_1, max_alt_type_1 = split_alteration_feature(max_pair_row["alteration_1"])
    max_gene_2, max_alt_type_2 = split_alteration_feature(max_pair_row["alteration_2"])

    summary = {
        "n_system_genes": len(genes),
        "n_features_tested": int(X_df.shape[1]),
        "n_pairs_candidate": n_pairs_candidate,
        "n_pairs_tested": n_pairs_tested,
        "n_pairs_skipped_low_pair_count": int(pair_df["model_status"].eq("skipped_low_pair_count").sum()),
        "n_pairs_skipped_sparse_combo": int(pair_df["model_status"].eq("skipped_sparse_combo").sum()),
        "n_pairs_failed": int(pair_df["model_status"].str.startswith("failed", na=False).sum()),
        "fdr_threshold": fdr_threshold,
        "n_epistasis_pairs_fdr": n_sig,
        "n_epistasis_pairs_fdr05": n_sig,
        "epistasis_score_fraction_fdr": n_sig / n_pairs_candidate,
        "epistasis_score_fraction_fdr05": n_sig / n_pairs_candidate,
        "epistasis_score_fraction_tested_fdr": n_sig / n_pairs_tested,
        "epistasis_score_mean_neglog10p": float(np.mean(-np.log10(pvals))),
        "epistasis_score_hmp_p": hmp,
        "epistasis_score_hmp_neglog10p": float(-np.log10(hmp)) if np.isfinite(hmp) else np.nan,
        "epistasis_score_max_neglog10p": float(np.max(-np.log10(pvals))),
        "max_neglog10p_pair": max_pair_row["pair"],
        "max_neglog10p_gene_1": max_gene_1,
        "max_neglog10p_alteration_type_1": max_alt_type_1,
        "max_neglog10p_gene_2": max_gene_2,
        "max_neglog10p_alteration_type_2": max_alt_type_2,
        "max_neglog10p_alteration_1": max_pair_row["alteration_1"],
        "max_neglog10p_alteration_2": max_pair_row["alteration_2"],
        "max_neglog10p_epistasis_p": float(max_pair_row["epistasis_p"]),
        "max_neglog10p_epistasis_fdr": float(max_pair_row["epistasis_fdr"]),
        "median_abs_interaction_beta": float(np.nanmedian(np.abs(fit_df["interaction_beta"].values))),
        "mean_delta_r2": float(np.nanmean(fit_df["delta_r2"].values)),
    }

    return summary, pair_df


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fast SciPy-free NeST system-level epistasis scoring with NumPy OLS and joblib."
    )

    parser.add_argument("--results_pkl", required=True)
    parser.add_argument("--system_to_genes_pkl", required=True)
    parser.add_argument("--drugs", required=True, help="Comma-separated drug names.")
    parser.add_argument("--response_key", default="actual", choices=["actual", "predictions"])

    parser.add_argument("--importance_csv", default="system_semi_partial_importance_g2d.csv")
    parser.add_argument("--crispr_csv", required=True, help="CSV with columns: system,best_fdr; optional drug column.")

    parser.add_argument("--amp_file", default="data/old_copynumber/cell2cnamplification_ctg_av.txt")
    parser.add_argument("--del_file", default="data/old_copynumber/cell2cndeletion_ctg_av.txt")
    parser.add_argument("--mut_file", default="data/cell2mutation_ctg_av.txt")
    parser.add_argument("--cell_index_file", default="data/cell2ind_av.txt")
    parser.add_argument("--gene_index_file", default="data/gene2ind_ctg_av.txt")

    parser.add_argument("--max_system_genes", type=int, default=100)
    parser.add_argument("--include_cnb", action="store_true", default=True)
    parser.add_argument("--no_cnb", action="store_false", dest="include_cnb")
    parser.add_argument("--include_tmb", action="store_true", default=False)

    parser.add_argument("--alteration_types", default="mut,amp,del")
    parser.add_argument("--include_same_gene_pairs", action="store_true", default=False)

    parser.add_argument("--min_alt_count", type=int, default=5)
    parser.add_argument("--min_pair_count", type=int, default=3)
    parser.add_argument("--min_cells_per_combo", type=int, default=2)
    parser.add_argument("--fdr_threshold", type=float, default=0.05)

    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--out_summary_csv", default="system_epistasis_scores_fast.csv")
    parser.add_argument("--out_pair_csv", default=None)

    args = parser.parse_args()

    drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    alteration_types = tuple(a.strip().lower() for a in args.alteration_types.split(",") if a.strip())

    results = load_pickle(args.results_pkl)
    system_to_genes = load_pickle(args.system_to_genes_pkl)

    importance_df = pd.read_csv(args.importance_csv)
    crispr_df = pd.read_csv(args.crispr_csv)

    cell_lines, genes = load_indices(args.cell_index_file, args.gene_index_file)

    amp_df = load_binary_df(args.amp_file, cell_lines, genes)
    del_df = load_binary_df(args.del_file, cell_lines, genes)
    mut_df = load_binary_df(args.mut_file, cell_lines, genes)

    systems_to_test = []
    for system, sys_genes in system_to_genes.items():
        indexed_genes = [g for g in sys_genes if g in genes]
        if 2 <= len(indexed_genes) <= args.max_system_genes:
            systems_to_test.append(system)

    print(f"Systems passing <= {args.max_system_genes} genes: {len(systems_to_test)}")
    print(f"Drugs: {drugs}")
    print(f"Response key: {args.response_key}")
    print(f"Include CNB: {args.include_cnb}")
    print(f"Include TMB: {args.include_tmb}")
    print(f"n_jobs: {args.n_jobs}")

    summary_records = []
    all_pair_dfs = []

    for drug in drugs:
        print(f"\n=== Drug: {drug} ===")
        y = get_drug_response(results, drug, args.response_key)

        for idx, system in enumerate(systems_to_test, start=1):
            if idx % 10 == 0:
                print(f"  processed {idx}/{len(systems_to_test)} systems")

            summary, pair_df = compute_system_epistasis_scores_fast(
                y=y,
                system=system,
                genes_in_system=system_to_genes[system],
                amp_df=amp_df,
                del_df=del_df,
                mut_df=mut_df,
                alteration_types=alteration_types,
                include_cnb=args.include_cnb,
                include_tmb=args.include_tmb,
                include_same_gene_pairs=args.include_same_gene_pairs,
                min_alt_count=args.min_alt_count,
                min_pair_count=args.min_pair_count,
                min_cells_per_combo=args.min_cells_per_combo,
                fdr_threshold=args.fdr_threshold,
                n_jobs=args.n_jobs,
            )

            importance_score, importance_p = lookup_importance(importance_df, drug, system)
            crispr_fdr = lookup_crispr_fdr(crispr_df, drug, system)

            if summary is None:
                summary = {
                    "n_system_genes": len([g for g in system_to_genes[system] if g in genes]),
                    "n_features_tested": 0,
                    "n_pairs_candidate": 0,
                    "n_pairs_tested": 0,
                    "n_pairs_skipped_low_pair_count": 0,
                    "n_pairs_skipped_sparse_combo": 0,
                    "n_pairs_failed": 0,
                    "fdr_threshold": args.fdr_threshold,
                    "n_epistasis_pairs_fdr": 0,
                    "n_epistasis_pairs_fdr05": 0,
                    "epistasis_score_fraction_fdr": np.nan,
                    "epistasis_score_fraction_fdr05": np.nan,
                    "epistasis_score_fraction_tested_fdr": np.nan,
                    "epistasis_score_mean_neglog10p": np.nan,
                    "epistasis_score_hmp_p": np.nan,
                    "epistasis_score_hmp_neglog10p": np.nan,
                    "epistasis_score_max_neglog10p": np.nan,
                    "max_neglog10p_pair": np.nan,
                    "max_neglog10p_gene_1": np.nan,
                    "max_neglog10p_alteration_type_1": np.nan,
                    "max_neglog10p_gene_2": np.nan,
                    "max_neglog10p_alteration_type_2": np.nan,
                    "max_neglog10p_alteration_1": np.nan,
                    "max_neglog10p_alteration_2": np.nan,
                    "max_neglog10p_epistasis_p": np.nan,
                    "max_neglog10p_epistasis_fdr": np.nan,
                    "median_abs_interaction_beta": np.nan,
                    "mean_delta_r2": np.nan,
                }

            summary_records.append({
                "drug": drug,
                "system": system,
                "drpt_importance_score": importance_score,
                "drpt_importance_p_bonferroni_pearson": importance_p,
                "crispr_best_fdr": crispr_fdr,
                "include_cnb": args.include_cnb,
                "include_tmb": args.include_tmb,
                **summary,
            })

            if args.out_pair_csv is not None and pair_df is not None and not pair_df.empty:
                pair_df = pair_df.copy()
                pair_df.insert(0, "drug", drug)
                all_pair_dfs.append(pair_df)

    summary_df = pd.DataFrame(summary_records)
    summary_df = add_hmp_fdr_by_drug(summary_df)

    summary_df = summary_df.sort_values(
        ["drug", "epistasis_score_hmp_fdr", "epistasis_score_fraction_fdr", "epistasis_score_mean_neglog10p"],
        ascending=[True, True, False, False],
        na_position="last"
    ).reset_index(drop=True)

    summary_df.to_csv(args.out_summary_csv, index=False)
    print(f"\nSaved summary: {args.out_summary_csv}")

    if args.out_pair_csv is not None:
        if len(all_pair_dfs) > 0:
            pair_out = pd.concat(all_pair_dfs, ignore_index=True)
            pair_out.to_csv(args.out_pair_csv, index=False)
            print(f"Saved pair-level results: {args.out_pair_csv}")
        else:
            print("No pair-level results to save.")


if __name__ == "__main__":
    main()