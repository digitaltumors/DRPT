#!/usr/bin/env python3

import argparse
import itertools
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from compute_system_epistasis import (
    Parallel,
    delayed,
    bh_fdr,
    get_drug_response,
    harmonic_mean_pvalue,
    load_binary_df,
    load_indices,
    split_alteration_feature,
    test_pair_fast,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_alteration_types(alteration_types):
    return tuple(a.strip().lower() for a in alteration_types.split(",") if a.strip())


def feature_name(gene: str, alteration_type: str) -> str:
    return f"{gene}_{alteration_type}"


def empty_gene_summary(gene, drug, n_systems_for_gene, fdr_threshold):
    return {
        "drug": drug,
        "gene": gene,
        "n_systems_for_gene": n_systems_for_gene,
        "n_partner_genes": 0,
        "n_features_for_gene": 0,
        "n_pairs_candidate": 0,
        "n_pairs_tested": 0,
        "n_pairs_skipped_low_pair_count": 0,
        "n_pairs_skipped_sparse_combo": 0,
        "n_pairs_failed": 0,
        "fdr_threshold": fdr_threshold,
        "n_epistasis_pairs_fdr": 0,
        "epistasis_score_fraction_fdr": np.nan,
        "epistasis_score_fraction_tested_fdr": np.nan,
        "epistasis_score_mean_neglog10p": np.nan,
        "epistasis_score_hmp_p": np.nan,
        "epistasis_score_hmp_neglog10p": np.nan,
        "epistasis_score_max_neglog10p": np.nan,
        "max_neglog10p_pair": np.nan,
        "max_neglog10p_partner_gene": np.nan,
        "max_neglog10p_gene_alteration": np.nan,
        "max_neglog10p_partner_alteration": np.nan,
        "max_neglog10p_epistasis_p": np.nan,
        "max_neglog10p_epistasis_fdr": np.nan,
        "median_abs_interaction_beta": np.nan,
        "mean_delta_r2": np.nan,
    }


def build_system_gene_maps(system_to_genes, panel_genes, max_system_genes):
    panel_gene_set = set(panel_genes)
    gene_to_systems = defaultdict(set)
    gene_pair_to_systems = defaultdict(set)
    systems_to_test = {}

    for system, sys_genes in system_to_genes.items():
        genes = sorted({str(g) for g in sys_genes if str(g) in panel_gene_set})
        if not (2 <= len(genes) <= max_system_genes):
            continue

        systems_to_test[system] = genes

        for gene in genes:
            gene_to_systems[gene].add(system)

        for gene1, gene2 in itertools.combinations(genes, 2):
            gene_pair_to_systems[(gene1, gene2)].add(system)

    return systems_to_test, gene_to_systems, gene_pair_to_systems


def build_feature_pairs(
    gene_pair_to_systems,
    gene_to_systems,
    alteration_types,
    include_same_gene_pairs=False,
):
    records = []

    for (gene1, gene2), systems in gene_pair_to_systems.items():
        for alt1 in alteration_types:
            for alt2 in alteration_types:
                f1 = feature_name(gene1, alt1)
                f2 = feature_name(gene2, alt2)
                records.append({
                    "gene_1": gene1,
                    "gene_2": gene2,
                    "alteration_1": f1,
                    "alteration_2": f2,
                    "systems": ";".join(sorted(systems)),
                    "n_systems_containing_gene_pair": len(systems),
                })

    if include_same_gene_pairs:
        genes = sorted(gene_to_systems)
        for gene in genes:
            for alt1, alt2 in itertools.combinations(alteration_types, 2):
                systems = gene_to_systems.get(gene, set())
                records.append({
                    "gene_1": gene,
                    "gene_2": gene,
                    "alteration_1": feature_name(gene, alt1),
                    "alteration_2": feature_name(gene, alt2),
                    "systems": ";".join(sorted(systems)),
                    "n_systems_containing_gene_pair": len(systems),
                })

    if not records:
        return pd.DataFrame(columns=[
            "gene_1",
            "gene_2",
            "alteration_1",
            "alteration_2",
            "systems",
            "n_systems_containing_gene_pair",
        ])

    return pd.DataFrame(records).drop_duplicates(["alteration_1", "alteration_2"]).reset_index(drop=True)


def build_feature_matrix(common, amp_sub, del_sub, mut_sub, alteration_types, min_alt_count):
    feature_series = {}
    feature_gene = {}

    for gene in amp_sub.columns:
        if "mut" in alteration_types:
            name = feature_name(gene, "mut")
            feature_series[name] = mut_sub[gene].astype(np.int8).values
            feature_gene[name] = gene
        if "amp" in alteration_types:
            name = feature_name(gene, "amp")
            feature_series[name] = amp_sub[gene].astype(np.int8).values
            feature_gene[name] = gene
        if "del" in alteration_types:
            name = feature_name(gene, "del")
            feature_series[name] = del_sub[gene].astype(np.int8).values
            feature_gene[name] = gene

    X_df = pd.DataFrame(feature_series, index=common)
    alt_counts = X_df.sum(axis=0)
    keep = alt_counts[alt_counts >= min_alt_count].index.tolist()
    return X_df[keep], feature_gene, alt_counts


def build_covariate_matrix(amp_sub, del_sub, mut_sub, include_cnb, include_tmb):
    covariates = []
    n_panel_genes = amp_sub.shape[1]

    if include_cnb:
        cnb = ((amp_sub.sum(axis=1).values + del_sub.sum(axis=1).values) / n_panel_genes).astype(float)
        covariates.append(cnb)

    if include_tmb:
        tmb = (mut_sub.sum(axis=1).values / n_panel_genes).astype(float)
        covariates.append(tmb)

    if covariates:
        return np.column_stack(covariates)
    return np.empty((amp_sub.shape[0], 0))


def run_pair_tests_for_drug(
    y,
    drug,
    feature_pairs_df,
    amp_df,
    del_df,
    mut_df,
    alteration_types,
    include_cnb,
    include_tmb,
    min_alt_count,
    min_pair_count,
    min_cells_per_combo,
    n_jobs,
):
    common = y.index.intersection(amp_df.index).intersection(del_df.index).intersection(mut_df.index)
    if len(common) < 10 or feature_pairs_df.empty:
        return pd.DataFrame()

    y_values = y.loc[common].values.astype(float)
    amp_sub = amp_df.loc[common]
    del_sub = del_df.loc[common]
    mut_sub = mut_df.loc[common]

    X_df, feature_gene, alt_counts = build_feature_matrix(
        common=common,
        amp_sub=amp_sub,
        del_sub=del_sub,
        mut_sub=mut_sub,
        alteration_types=alteration_types,
        min_alt_count=min_alt_count,
    )

    if X_df.shape[1] < 2:
        return pd.DataFrame()

    candidate_df = feature_pairs_df[
        feature_pairs_df["alteration_1"].isin(X_df.columns)
        & feature_pairs_df["alteration_2"].isin(X_df.columns)
    ].copy()

    if candidate_df.empty:
        return pd.DataFrame()

    feature_names = list(X_df.columns)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    feature_gene_list = [feature_gene[f] for f in feature_names]
    X_values = X_df.values.astype(np.int8)
    covariate_matrix = build_covariate_matrix(amp_sub, del_sub, mut_sub, include_cnb, include_tmb)

    test_inputs = [
        (feature_index[row.alteration_1], feature_index[row.alteration_2])
        for row in candidate_df.itertuples(index=False)
    ]

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
                include_same_gene_pairs=True,
                min_pair_count=min_pair_count,
                min_cells_per_combo=min_cells_per_combo,
            )
            for i1, i2 in test_inputs
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
                include_same_gene_pairs=True,
                min_pair_count=min_pair_count,
                min_cells_per_combo=min_cells_per_combo,
            )
            for i1, i2 in test_inputs
        )

    pair_df = pd.DataFrame(pair_results)
    if pair_df.empty:
        return pair_df

    pair_df.insert(0, "drug", drug)
    pair_df["epistasis_fdr"] = bh_fdr(pair_df["epistasis_p"].values)
    pair_df[["gene_1", "alteration_type_1"]] = pair_df["alteration_1"].apply(
        lambda x: pd.Series(split_alteration_feature(x))
    )
    pair_df[["gene_2", "alteration_type_2"]] = pair_df["alteration_2"].apply(
        lambda x: pd.Series(split_alteration_feature(x))
    )
    pair_df["alteration_1_count"] = pair_df["alteration_1"].map(alt_counts).astype(float)
    pair_df["alteration_2_count"] = pair_df["alteration_2"].map(alt_counts).astype(float)

    metadata_cols = [
        "alteration_1",
        "alteration_2",
        "systems",
        "n_systems_containing_gene_pair",
    ]
    pair_df = pair_df.merge(candidate_df[metadata_cols], on=["alteration_1", "alteration_2"], how="left")
    return pair_df


def summarize_gene(drug, gene, pair_df, gene_to_systems, fdr_threshold):
    summary = empty_gene_summary(
        gene=gene,
        drug=drug,
        n_systems_for_gene=len(gene_to_systems.get(gene, [])),
        fdr_threshold=fdr_threshold,
    )

    if pair_df.empty or "gene_1" not in pair_df.columns or "gene_2" not in pair_df.columns:
        return summary

    sub = pair_df[(pair_df["gene_1"].eq(gene)) | (pair_df["gene_2"].eq(gene))].copy()
    if sub.empty:
        return summary

    partner_genes = set(sub.loc[sub["gene_1"].eq(gene), "gene_2"])
    partner_genes.update(set(sub.loc[sub["gene_2"].eq(gene), "gene_1"]))
    partner_genes.discard(gene)

    gene_features = set(sub.loc[sub["gene_1"].eq(gene), "alteration_1"])
    gene_features.update(set(sub.loc[sub["gene_2"].eq(gene), "alteration_2"]))

    summary.update({
        "n_partner_genes": len(partner_genes),
        "n_features_for_gene": len(gene_features),
        "n_pairs_candidate": int(sub.shape[0]),
        "n_pairs_skipped_low_pair_count": int(sub["model_status"].eq("skipped_low_pair_count").sum()),
        "n_pairs_skipped_sparse_combo": int(sub["model_status"].eq("skipped_sparse_combo").sum()),
        "n_pairs_failed": int(sub["model_status"].str.startswith("failed", na=False).sum()),
    })

    fit_df = sub[sub["model_status"].eq("fit") & sub["epistasis_p"].notna()].copy()
    if fit_df.empty:
        summary["epistasis_score_fraction_fdr"] = 0.0
        return summary

    pvals = fit_df["epistasis_p"].clip(lower=np.nextafter(0, 1)).values
    fdrs = fit_df["epistasis_fdr"].values
    hmp = harmonic_mean_pvalue(pvals)
    n_sig = int(np.sum(fdrs < fdr_threshold))
    max_pair_row = fit_df.loc[fit_df["epistasis_p"].idxmin()]

    if max_pair_row["gene_1"] == gene:
        partner_gene = max_pair_row["gene_2"]
        gene_alteration = max_pair_row["alteration_1"]
        partner_alteration = max_pair_row["alteration_2"]
    else:
        partner_gene = max_pair_row["gene_1"]
        gene_alteration = max_pair_row["alteration_2"]
        partner_alteration = max_pair_row["alteration_1"]

    summary.update({
        "n_pairs_tested": int(fit_df.shape[0]),
        "n_epistasis_pairs_fdr": n_sig,
        "epistasis_score_fraction_fdr": n_sig / int(sub.shape[0]),
        "epistasis_score_fraction_tested_fdr": n_sig / int(fit_df.shape[0]),
        "epistasis_score_mean_neglog10p": float(np.mean(-np.log10(pvals))),
        "epistasis_score_hmp_p": hmp,
        "epistasis_score_hmp_neglog10p": float(-np.log10(hmp)) if np.isfinite(hmp) else np.nan,
        "epistasis_score_max_neglog10p": float(np.max(-np.log10(pvals))),
        "max_neglog10p_pair": max_pair_row["pair"],
        "max_neglog10p_partner_gene": partner_gene,
        "max_neglog10p_gene_alteration": gene_alteration,
        "max_neglog10p_partner_alteration": partner_alteration,
        "max_neglog10p_epistasis_p": float(max_pair_row["epistasis_p"]),
        "max_neglog10p_epistasis_fdr": float(max_pair_row["epistasis_fdr"]),
        "median_abs_interaction_beta": float(np.nanmedian(np.abs(fit_df["interaction_beta"].values))),
        "mean_delta_r2": float(np.nanmean(fit_df["delta_r2"].values)),
    })
    return summary


def add_gene_hmp_fdr_by_drug(summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = summary_df.copy()
    summary_df["epistasis_score_hmp_fdr"] = np.nan
    summary_df["epistasis_score_hmp_qvalue"] = np.nan
    summary_df["epistasis_score_hmp_neglog10q"] = np.nan
    summary_df["epistasis_score_hmp_fdr_n_genes"] = 0

    if summary_df.empty or "epistasis_score_hmp_p" not in summary_df.columns:
        return summary_df

    for drug, idx in summary_df.groupby("drug", sort=False).groups.items():
        hmp_pvals = summary_df.loc[idx, "epistasis_score_hmp_p"].values
        qvals = bh_fdr(hmp_pvals)
        n_valid = int(np.isfinite(hmp_pvals).sum())
        summary_df.loc[idx, "epistasis_score_hmp_fdr"] = qvals
        summary_df.loc[idx, "epistasis_score_hmp_qvalue"] = qvals
        summary_df.loc[idx, "epistasis_score_hmp_fdr_n_genes"] = n_valid

    qvals = summary_df["epistasis_score_hmp_qvalue"].clip(lower=np.nextafter(0, 1))
    valid = np.isfinite(qvals)
    summary_df.loc[valid, "epistasis_score_hmp_neglog10q"] = -np.log10(qvals[valid])
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Gene-level epistasis scoring from deduplicated alteration pairs observed within NeST systems."
    )
    parser.add_argument("--results_pkl", required=True)
    parser.add_argument("--system_to_genes_pkl", required=True)
    parser.add_argument("--drugs", required=True, help="Comma-separated drug names.")
    parser.add_argument("--response_key", default="actual", choices=["actual", "predictions"])

    parser.add_argument("--amp_file", default=str(REPOSITORY_ROOT / "data/cell2cnamplification_ctg_av.txt"))
    parser.add_argument("--del_file", default=str(REPOSITORY_ROOT / "data/cell2cndeletion_ctg_av.txt"))
    parser.add_argument("--mut_file", default=str(REPOSITORY_ROOT / "data/cell2mutation_ctg_av.txt"))
    parser.add_argument("--cell_index_file", default=str(REPOSITORY_ROOT / "data/cell2ind_av.txt"))
    parser.add_argument("--gene_index_file", default=str(REPOSITORY_ROOT / "data/gene2ind_ctg_av.txt"))

    parser.add_argument("--max_system_genes", type=int, default=100)
    parser.add_argument("--include_cnb", action="store_true", default=True)
    parser.add_argument("--no_cnb", action="store_false", dest="include_cnb")
    parser.add_argument("--include_tmb", action="store_true", default=False)
    parser.add_argument("--alteration_types", default="mut,amp,del")
    parser.add_argument("--include_same_gene_pairs", action="store_true", default=False)

    parser.add_argument("--min_alt_count", type=int, default=5)
    parser.add_argument("--min_pair_count", type=int, default=10)
    parser.add_argument("--min_cells_per_combo", type=int, default=5)
    parser.add_argument("--fdr_threshold", type=float, default=0.1)
    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--out_gene_summary_csv", default="gene_epistasis_scores.csv")
    parser.add_argument("--out_pair_csv", default=None)

    args = parser.parse_args()

    drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    alteration_types = normalize_alteration_types(args.alteration_types)

    results = load_pickle(args.results_pkl)
    system_to_genes = load_pickle(args.system_to_genes_pkl)
    cell_lines, genes = load_indices(args.cell_index_file, args.gene_index_file)

    amp_df = load_binary_df(args.amp_file, cell_lines, genes)
    del_df = load_binary_df(args.del_file, cell_lines, genes)
    mut_df = load_binary_df(args.mut_file, cell_lines, genes)

    systems_to_test, gene_to_systems, gene_pair_to_systems = build_system_gene_maps(
        system_to_genes=system_to_genes,
        panel_genes=genes,
        max_system_genes=args.max_system_genes,
    )
    feature_pairs_df = build_feature_pairs(
        gene_pair_to_systems=gene_pair_to_systems,
        gene_to_systems=gene_to_systems,
        alteration_types=alteration_types,
        include_same_gene_pairs=args.include_same_gene_pairs,
    )

    genes_to_score = sorted(gene_to_systems)
    print(f"Systems passing <= {args.max_system_genes} genes: {len(systems_to_test)}")
    print(f"Genes represented in eligible systems: {len(genes_to_score)}")
    print(f"Unique gene pairs from eligible systems: {len(gene_pair_to_systems)}")
    print(f"Unique alteration pairs to consider before abundance filters: {len(feature_pairs_df)}")
    print(f"Drugs: {drugs}")
    print(f"Include CNB: {args.include_cnb}")
    print(f"Include TMB: {args.include_tmb}")
    print(f"n_jobs: {args.n_jobs}")

    summary_records = []
    all_pair_dfs = []

    for drug in drugs:
        print(f"\n=== Drug: {drug} ===")
        y = get_drug_response(results, drug, args.response_key)
        pair_df = run_pair_tests_for_drug(
            y=y,
            drug=drug,
            feature_pairs_df=feature_pairs_df,
            amp_df=amp_df,
            del_df=del_df,
            mut_df=mut_df,
            alteration_types=alteration_types,
            include_cnb=args.include_cnb,
            include_tmb=args.include_tmb,
            min_alt_count=args.min_alt_count,
            min_pair_count=args.min_pair_count,
            min_cells_per_combo=args.min_cells_per_combo,
            n_jobs=args.n_jobs,
        )

        print(f"  unique alteration pairs retained after abundance filters: {len(pair_df)}")

        for idx, gene in enumerate(genes_to_score, start=1):
            if idx % 100 == 0:
                print(f"  summarized {idx}/{len(genes_to_score)} genes")

            summary_records.append(summarize_gene(
                drug=drug,
                gene=gene,
                pair_df=pair_df,
                gene_to_systems=gene_to_systems,
                fdr_threshold=args.fdr_threshold,
            ))

        if args.out_pair_csv is not None and pair_df is not None and not pair_df.empty:
            all_pair_dfs.append(pair_df)

    summary_df = pd.DataFrame(summary_records)
    summary_df = add_gene_hmp_fdr_by_drug(summary_df)
    summary_df = summary_df.sort_values(
        ["drug", "epistasis_score_hmp_fdr", "epistasis_score_fraction_fdr", "epistasis_score_mean_neglog10p"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    summary_df.to_csv(args.out_gene_summary_csv, index=False)
    print(f"\nSaved gene summary: {args.out_gene_summary_csv}")

    if args.out_pair_csv is not None:
        if all_pair_dfs:
            pair_out = pd.concat(all_pair_dfs, ignore_index=True)
            pair_out.to_csv(args.out_pair_csv, index=False)
            print(f"Saved deduplicated pair-level results: {args.out_pair_csv}")
        else:
            print("No pair-level results to save.")


if __name__ == "__main__":
    main()
