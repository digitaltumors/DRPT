#!/usr/bin/env python3

"""
Generalized DRPT -> patient transfer pipeline for MSK-CHORD RSI cohorts.

This script extends the single-drug example in drpt_patient_transfer.py to:
1. work with any number of RSI drugs present in msk_patients_rsi_drugs.txt,
2. split by unique PATIENT_ID for nested 60/20/20 train/val/test evaluation,
3. optionally add configurable patient covariates on top of predicted AUC,
4. support an AUC-only mode with no survival fine-tuning,
5. emit row-aligned outputs for downstream survival stratification.

The DRPT-specific model construction is intentionally delegated to a bootstrap
module supplied with --drpt-bootstrap. That module must expose a function:

    build_drpt_components(
        cell_line_model: str,
        cell_line_response: str,
        cell_gene2ind: str,
        ontology: str,
        hidden_dims: int,
        dropout: float,
        diff_transformer: bool,
        device: str,
    ) -> dict

and return a dict with keys:
    - compound_encoder
    - tree_parser
    - cell_response_model

See drpt_bootstrap_template.py for a starting point.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtuples as tt
from lifelines.statistics import logrank_test
from pycox.models import CoxPH, CoxTime, LogisticHazard, PCHazard
from pycox.models.cox_time import MLPVanillaCoxTime
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SMILES_TO_NAME = {
    "C1=C(C(=O)NC(=O)N1)F": "5-FU",
    "C1=CN(C(=O)N=C1N)[C@H]2C([C@@H]([C@H](O2)CO)O)(F)F": "Gemcitabine",
    "C1C2CC3CC1CC(C2)(C3)C4=C(C=CC(=C4)C5=CC6=C(C=C5)C=C(C=C6)C(=O)O)O": "CD437",
    "C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC(=C3)CC4=NNC(=O)C5=CC=CC=C54)F": "Ceralasertib",
    "CC(C)(C1=NC(=CC=C1)N2C3=NC(=NC=C3C(=O)N2CC=C)NC4=CC=C(C=C4)N5CCN(CC5)C)O": "MK-1775",
    "CC1=C(N=C(N=C1N)[C@H](CC(=O)N)NC[C@@H](C(=O)N)N)C(=O)N[C@@H]([C@H](C2=CN=CN2)O[C@H]3[C@H]([C@H]([C@@H]([C@@H](O3)CO)O)O)O[C@@H]4[C@H]([C@H]([C@@H]([C@H](O4)CO)O)OC(=O)N)O)C(=O)N[C@H](C)[C@H]([C@H](C)C(=O)N[C@@H]([C@@H](C)O)C(=O)NCCC5=NC(=CS5)C6=NC(=CS6)C(=O)NCCC[S+](C)C)O": "Bleomycin-a2",
    "CC[C@@]1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O": "Camptothecin",
    "CN(CC1=CN=C2C(=N1)C(=NC(=N2)N)N)C3=CC=C(C=C3)C(=O)N[C@@H](CCC(=O)O)C(=O)O": "Methotrexate",
    "C[C@@H]1COCCN1C2=NC(=NC(=C2)C3(CC3)[S@](=N)(=O)C)C4=C5C=CNC5=NC=C4": "Olaparib",
    "C[C@@H]1OC[C@@H]2[C@@H](O1)[C@@H]([C@H]([C@@H](O2)O[C@H]3[C@H]4COC(=O)[C@@H]4[C@@H](C5=CC6=C(C=C35)OCO6)C7=CC(=C(C(=C7)OC)O)OC)O)O": "Etoposide",
    "C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O": "Doxorubicin",
    "N.N.[Cl-].[Cl-].[Pt+2]": "Cisplatin",
}
NAME_TO_SMILES = {value: key for key, value in SMILES_TO_NAME.items()}


MANDATORY_FEATURES = ["sex", "age"]
OPTIONAL_COVARIATES = {
    "stage": {"column": "STAGE_HIGHEST_RECORDED", "type": "categorical"},
    "smoking_status": {"column": "SMOKING_PREDICTIONS_3_CLASSES", "type": "categorical"},
    "msi_score": {"column": "MSI_SCORE", "type": "numeric"},
    "msi_type": {"column": "MSI_TYPE", "type": "categorical"},
    "sample_type": {"column": "SAMPLE_TYPE", "type": "categorical"},
    "rsi_start_date": {"column": "RSI_START_DATE", "type": "numeric"},
    "cancer_type": {"column": "CANCER_TYPE", "type": "categorical"},
    "cancer_type_detailed": {"column": "CANCER_TYPE_DETAILED", "type": "categorical"},
    "radiation_therapy": {"column": "RADIATION_THERAPY", "type": "categorical"},
}


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to(subvalue, device) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [move_to(subvalue, device) for subvalue in value]
        return type(value)(moved)
    return value


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        # Compatibility with older scikit-learn versions.
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def load_bootstrap_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("drpt_bootstrap_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load DRPT bootstrap module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_drpt_components"):
        raise RuntimeError(
            f"{module_path} must define build_drpt_components(...). "
            "See drpt_bootstrap_template.py for the expected interface."
        )
    return module


def load_binary_matrix(path: Path) -> np.ndarray:
    rows: List[np.ndarray] = []
    expected_len: Optional[int] = None
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue

            raw = row[0].strip().strip('"')
            if not raw:
                continue

            # Allow accidental single-column headers such as "mutation".
            if "," not in raw and not raw.lstrip("-").isdigit():
                continue

            tokens = [token.strip() for token in raw.split(",")]
            try:
                values = np.asarray([int(float(token)) for token in tokens], dtype=np.int32)
            except ValueError as exc:
                raise ValueError(f"Unable to parse binary row {line_number} from {path}: {raw[:100]}") from exc

            if expected_len is None:
                expected_len = len(values)
            elif len(values) != expected_len:
                raise ValueError(
                    f"Inconsistent vector length in {path} at line {line_number}: "
                    f"expected {expected_len}, found {len(values)}"
                )

            rows.append(values)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return np.vstack(rows)


def load_patient_index(path: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) >= 2 and row[0].strip().lower() == "index" and row[1].strip().lower() == "sample":
                continue
            mapping[row[1]] = int(row[0])
    return mapping


def encode_sex(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {"male": 1.0, "female": 0.0}
    return cleaned.map(mapping)


def encode_event(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split(":").str[0].astype(int)


def normalize_unknown_to_sentinel(series: pd.Series, sentinel: str = "-1") -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    return cleaned.mask(cleaned.eq("") | cleaned.str.lower().eq("unknown"), sentinel)


def resolve_lookup_sample_id(sample_id: str, sample2ind: Dict[str, int]) -> Optional[str]:
    sample_id = str(sample_id).strip()
    if not sample_id:
        return None

    candidates = [sample_id]
    if not sample_id.endswith("-01"):
        candidates.append(sample_id + "-01")
    if sample_id.endswith("-01"):
        candidates.append(sample_id[:-3])

    for candidate in candidates:
        if candidate in sample2ind:
            return candidate
    return None


def ensure_column(df: pd.DataFrame, target: str, aliases: Sequence[str], default_value="") -> None:
    if target in df.columns:
        return
    for alias in aliases:
        if alias in df.columns:
            df[target] = df[alias]
            return
    df[target] = default_value


def normalize_patient_table(patient_df: pd.DataFrame, allowed_drugs: Optional[Sequence[str]]) -> pd.DataFrame:
    df = patient_df.copy()
    ensure_column(df, "CURRENT_AGE_DEID", ["AGE"], default_value=np.nan)
    ensure_column(df, "STAGE_HIGHEST_RECORDED", ["CLINICAL_STAGE", "PATHOLOGIC_STAGE"], default_value="")
    ensure_column(df, "SMOKING_PREDICTIONS_3_CLASSES", [], default_value="")
    ensure_column(df, "MSI_SCORE", [], default_value=np.nan)
    ensure_column(df, "MSI_TYPE", [], default_value="")
    ensure_column(df, "SAMPLE_TYPE", [], default_value="")
    ensure_column(df, "RSI_START_DATE", [], default_value=np.nan)
    ensure_column(df, "CANCER_TYPE_DETAILED", [], default_value="")
    ensure_column(df, "RADIATION_THERAPY", [], default_value="")

    df["RSI_DRUG"] = df["RSI_DRUG"].astype(str).str.strip()
    if allowed_drugs:
        allowed = {drug.strip() for drug in allowed_drugs}
        df = df[df["RSI_DRUG"].isin(allowed)].copy()

    df = df[df["RSI_DRUG"].isin(NAME_TO_SMILES)].copy()
    df["DRUG_SMILES"] = df["RSI_DRUG"].map(NAME_TO_SMILES)
    df["SEX_ENCODED"] = encode_sex(df["GENDER"])
    df["EVENT"] = encode_event(df["PFS_STATUS"])
    df["PFS_MONTHS"] = pd.to_numeric(df["PFS_MONTHS"], errors="coerce")
    df["CURRENT_AGE_DEID"] = pd.to_numeric(df["CURRENT_AGE_DEID"], errors="coerce")
    df["RSI_START_DATE"] = pd.to_numeric(df["RSI_START_DATE"], errors="coerce")
    df["MSI_SCORE"] = pd.to_numeric(df["MSI_SCORE"], errors="coerce")
    df["SMOKING_PREDICTIONS_3_CLASSES"] = normalize_unknown_to_sentinel(df["SMOKING_PREDICTIONS_3_CLASSES"])
    df["SAMPLE_TYPE"] = normalize_unknown_to_sentinel(df["SAMPLE_TYPE"])

    df = df.dropna(subset=["PATIENT_ID", "SAMPLE_ID", "DRUG_SMILES", "SEX_ENCODED", "CURRENT_AGE_DEID", "PFS_MONTHS", "EVENT"])
    df = df[df["PFS_MONTHS"] >= 0].copy()
    df = df.reset_index(drop=True)
    df["ROW_ID"] = np.arange(len(df))
    return df


class PatientSampleDataset(Dataset):
    def __init__(
        self,
        response_df: pd.DataFrame,
        sample2ind: Dict[str, int],
        genotype_arrays: Dict[str, np.ndarray],
        compound_encoder,
        tree_parser,
        mut2gene: bool = True,
        with_indices: bool = True,
    ):
        self.response_df = response_df.reset_index(drop=True)
        self.sample2ind = sample2ind
        self.genotype_arrays = genotype_arrays
        self.compound_encoder = compound_encoder
        self.tree_parser = tree_parser
        self.mut2gene = mut2gene
        self.with_indices = with_indices
        self.drug_dict = {drug: self.compound_encoder.encode(drug) for drug in self.response_df["DRUG_SMILES"].unique()}

    def __len__(self):
        return len(self.response_df)

    def __getitem__(self, index):
        row = self.response_df.iloc[index]
        lookup_sample_id = row["LOOKUP_SAMPLE_ID"] if "LOOKUP_SAMPLE_ID" in row.index else row["SAMPLE_ID"]
        sample_index = self.sample2ind[lookup_sample_id]

        if self.with_indices:
            if self.mut2gene:
                patient_mut_dict = {
                    mut_type: self.tree_parser.get_mut2gene(
                        np.where(array[sample_index] == 1)[0],
                        type_indices={1.0: np.where(array[sample_index] == 1)[0]},
                    )
                    for mut_type, array in self.genotype_arrays.items()
                }
            else:
                patient_mut_dict = {
                    mut_type: self.tree_parser.get_mut2sys(
                        np.where(array[sample_index] == 1)[0],
                        type_indices={1.0: np.where(array[sample_index] == 1)[0]},
                    )
                    for mut_type, array in self.genotype_arrays.items()
                }
        else:
            if self.mut2gene:
                patient_mut_dict = {
                    mut_type: self.tree_parser.get_mutation2genotype_mask(
                        torch.tensor(array[sample_index], dtype=torch.float32)
                    )
                    for mut_type, array in self.genotype_arrays.items()
                }
            else:
                patient_mut_dict = {
                    mut_type: self.tree_parser.get_system2genotype_mask(
                        torch.tensor(array[sample_index], dtype=torch.float32)
                    )
                    for mut_type, array in self.genotype_arrays.items()
                }

        return {
            "row_id": int(row["ROW_ID"]),
            "genotype": patient_mut_dict,
            "drug": self.drug_dict[row["DRUG_SMILES"]],
        }


class PatientSampleCollator:
    def __init__(self, tree_parser, genotypes, compound_encoder, mut2gene: bool = True, with_indices: bool = True):
        self.tree_parser = tree_parser
        self.genotypes = genotypes
        self.compound_encoder = compound_encoder
        self.mut2gene = mut2gene
        self.with_indices = with_indices

    def __call__(self, batch):
        mutation_dict = {}
        for genotype in self.genotypes:
            if self.with_indices:
                embedding_dict = {}
                if self.mut2gene:
                    embedding_dict["mut"] = pad_sequence(
                        [item["genotype"][genotype]["mut"] for item in batch],
                        padding_value=self.tree_parser.n_genes - 1,
                        batch_first=True,
                    ).to(torch.long)
                    max_len = embedding_dict["mut"].size(1)
                    embedding_dict["mask"] = torch.stack(
                        [item["genotype"][genotype]["mask"] for item in batch]
                    )[:, :max_len, :max_len]
                else:
                    embedding_dict["gene"] = pad_sequence(
                        [item["genotype"][genotype]["gene"] for item in batch],
                        padding_value=self.tree_parser.n_genes,
                        batch_first=True,
                    ).to(torch.long)
                    embedding_dict["sys"] = pad_sequence(
                        [item["genotype"][genotype]["sys"] for item in batch],
                        padding_value=self.tree_parser.n_systems,
                        batch_first=True,
                    ).to(torch.long)
                    gene_max_len = embedding_dict["gene"].size(1)
                    sys_max_len = embedding_dict["sys"].size(1)
                    embedding_dict["mask"] = torch.stack(
                        [item["genotype"][genotype]["mask"] for item in batch]
                    )[:, :sys_max_len, :gene_max_len]
                mutation_dict[genotype] = embedding_dict
            else:
                mutation_dict[genotype] = torch.stack([item["genotype"][genotype] for item in batch])

        return {
            "row_id": torch.tensor([item["row_id"] for item in batch], dtype=torch.long),
            "genotype": mutation_dict,
            "drug": self.compound_encoder.collate([item["drug"] for item in batch]),
        }


def _prepare_drpt_batch_artifacts(dataset: PatientSampleDataset, device: str):
    nested_subtrees_forward = move_to(dataset.tree_parser.get_nested_subtree_mask(["default"], direction="forward"), device)
    nested_subtrees_backward = move_to(dataset.tree_parser.get_nested_subtree_mask(["default"], direction="backward"), device)
    gene2system_mask = move_to(torch.tensor(dataset.tree_parser.gene2sys_mask, dtype=torch.bool), device)
    system2gene_mask = move_to(torch.tensor(dataset.tree_parser.sys2gene_mask, dtype=torch.bool), device)
    return nested_subtrees_forward, nested_subtrees_backward, gene2system_mask, system2gene_mask


def _compute_drpt_embeddings(
    cell_response_model,
    batch,
    nested_subtrees_forward,
    nested_subtrees_backward,
    gene2system_mask,
    system2gene_mask,
    sys2gene: bool,
):
    compound = batch["drug"]
    genotype_dict = batch["genotype"]
    batch_size = compound.size(0)

    gene_embedding = cell_response_model.get_mut2gene(genotype_dict, with_indices=True, batch_size=batch_size)
    system_embedding = cell_response_model.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
    system_embedding, gene_effect = cell_response_model.get_gene2sys(system_embedding, gene_embedding, gene2system_mask)
    system_embedding = system_embedding + cell_response_model.effect_norm(gene_effect)

    system_embedding = cell_response_model.get_sys2sys(
        system_embedding,
        nested_subtrees_forward,
        direction="forward",
        return_updates=False,
        with_indices=False,
    )
    system_embedding = cell_response_model.get_sys2sys(
        system_embedding,
        nested_subtrees_backward,
        direction="backward",
        return_updates=False,
        with_indices=False,
    )

    if sys2gene:
        gene_embedding, system_effect_on_gene = cell_response_model.get_sys2gene(
            gene_embedding,
            system_embedding,
            system2gene_mask,
        )
        gene_embedding = gene_embedding + cell_response_model.effect_norm(system_effect_on_gene)

    compound_embedding = cell_response_model.get_compound_embedding(compound, unsqueeze=True)
    drug_system_embedding = cell_response_model.get_system2comp(compound_embedding, system_embedding, attention=False, score=False)
    drug_gene_embedding = cell_response_model.get_gene2comp(compound_embedding, gene_embedding, attention=False, score=False)

    drug_system_embedding = drug_system_embedding.squeeze(1)
    drug_gene_embedding = drug_gene_embedding.squeeze(1)
    return torch.cat([drug_system_embedding, drug_gene_embedding], dim=-1)


def predict_auc_for_all_rows(
    response_df: pd.DataFrame,
    sample2ind: Dict[str, int],
    genotype_arrays: Dict[str, np.ndarray],
    compound_encoder,
    tree_parser,
    cell_response_model,
    device: str,
    jobs: int,
    auc_batch_size: int,
    sys2gene: bool,
    gene2drug: bool,
) -> pd.Series:
    if response_df.empty:
        raise ValueError(
            "No patient rows are available for DRPT AUC prediction. "
            "Check your patient table filtering and SAMPLE_ID overlap with patient2ind."
        )

    dataset = PatientSampleDataset(
        response_df=response_df,
        sample2ind=sample2ind,
        genotype_arrays=genotype_arrays,
        compound_encoder=compound_encoder,
        tree_parser=tree_parser,
        mut2gene=True,
        with_indices=True,
    )
    collator = PatientSampleCollator(tree_parser, list(genotype_arrays.keys()), compound_encoder, mut2gene=True, with_indices=True)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=min(max(1, auc_batch_size), max(1, len(dataset))),
        num_workers=max(0, jobs),
        collate_fn=collator,
    )

    nested_subtrees_forward = move_to(dataset.tree_parser.get_nested_subtree_mask(["default"], direction="forward"), device)
    nested_subtrees_backward = move_to(dataset.tree_parser.get_nested_subtree_mask(["default"], direction="backward"), device)
    gene2system_mask = move_to(torch.tensor(dataset.tree_parser.gene2sys_mask, dtype=torch.bool), device)
    system2gene_mask = move_to(torch.tensor(dataset.tree_parser.sys2gene_mask, dtype=torch.bool), device)

    cell_response_model.eval()
    auc_by_row: Dict[int, float] = {}
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to(batch, device)
            auc = cell_response_model(
                batch["genotype"],
                batch["drug"],
                nested_subtrees_forward,
                nested_subtrees_backward,
                gene2system_mask,
                system2gene_mask,
                sys2cell=True,
                cell2sys=True,
                sys2gene=sys2gene,
                gene2drug=gene2drug,
                mut2gene=True,
                with_indices=True,
            )
            auc = auc.squeeze().detach().cpu().numpy()
            if np.isscalar(auc):
                auc = np.asarray([float(auc)])
            row_ids = batch["row_id"].detach().cpu().numpy()
            for row_id, auc_value in zip(row_ids, auc):
                auc_by_row[int(row_id)] = float(auc_value)

    auc_series = pd.Series(auc_by_row, name="predicted_auc")
    return auc_series.reindex(response_df["ROW_ID"]).reset_index(drop=True)


def predict_embedding_features_for_all_rows(
    response_df: pd.DataFrame,
    sample2ind: Dict[str, int],
    genotype_arrays: Dict[str, np.ndarray],
    compound_encoder,
    tree_parser,
    cell_response_model,
    device: str,
    jobs: int,
    auc_batch_size: int,
    sys2gene: bool,
) -> pd.DataFrame:
    if response_df.empty:
        raise ValueError(
            "No patient rows are available for embedding transfer. "
            "Check your patient table filtering and SAMPLE_ID overlap with patient2ind."
        )

    dataset = PatientSampleDataset(
        response_df=response_df,
        sample2ind=sample2ind,
        genotype_arrays=genotype_arrays,
        compound_encoder=compound_encoder,
        tree_parser=tree_parser,
        mut2gene=True,
        with_indices=True,
    )
    collator = PatientSampleCollator(tree_parser, list(genotype_arrays.keys()), compound_encoder, mut2gene=True, with_indices=True)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=min(max(1, auc_batch_size), max(1, len(dataset))),
        num_workers=max(0, jobs),
        collate_fn=collator,
    )

    nested_subtrees_forward, nested_subtrees_backward, gene2system_mask, system2gene_mask = _prepare_drpt_batch_artifacts(dataset, device)

    embedding_by_row: Dict[int, np.ndarray] = {}
    cell_response_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = move_to(batch, device)
            embeddings = _compute_drpt_embeddings(
                cell_response_model,
                batch,
                nested_subtrees_forward,
                nested_subtrees_backward,
                gene2system_mask,
                system2gene_mask,
                sys2gene=sys2gene,
            )
            embeddings = embeddings.detach().cpu().numpy()
            row_ids = batch["row_id"].detach().cpu().numpy()
            for row_id, embedding in zip(row_ids, embeddings):
                embedding_by_row[int(row_id)] = embedding.astype(np.float32, copy=False)

    ordered = [embedding_by_row[int(row_id)] for row_id in response_df["ROW_ID"].values]
    if not ordered:
        raise ValueError(
            "Embedding extraction produced zero rows. "
            "This usually means the filtered patient cohort is empty after matching SAMPLE_ID values to patient2ind."
        )
    embedding_matrix = np.vstack(ordered)
    columns = [f"embedding_{idx}" for idx in range(embedding_matrix.shape[1])]
    return pd.DataFrame(embedding_matrix, columns=columns)


@dataclass
class FeatureArtifacts:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    trainval_transformer: Optional[Dict[str, object]]
    x_trainval: np.ndarray
    x_test_final: np.ndarray
    feature_names: List[str]


class LayerNormLinearHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=0.1)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(self.norm(x))


def build_feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trainval_df: pd.DataFrame,
    extra_covariates: Sequence[str],
    include_drug: bool,
    include_auc: bool,
) -> FeatureArtifacts:
    numeric_cols = [("age", "CURRENT_AGE_DEID")]
    if include_auc:
        numeric_cols.insert(0, ("predicted_auc", "predicted_auc"))
    categorical_cols = [("sex", "SEX_ENCODED")]

    for covariate in extra_covariates:
        spec = OPTIONAL_COVARIATES[covariate]
        if spec["type"] == "numeric":
            numeric_cols.append((covariate, spec["column"]))
        else:
            categorical_cols.append((covariate, spec["column"]))

    if include_drug:
        categorical_cols.append(("drug", "RSI_DRUG"))

    def fit_transformer(df: pd.DataFrame):
        scaler = StandardScaler()
        x_numeric = np.empty((len(df), 0), dtype=np.float32)
        numeric_feature_names: List[str] = []
        if numeric_cols:
            numeric_values = df[[column for _, column in numeric_cols]].astype(float).fillna(0.0).values
            x_numeric = scaler.fit_transform(numeric_values).astype(np.float32)
            numeric_feature_names = [name for name, _ in numeric_cols]

        encoder = None
        x_categorical = np.empty((len(df), 0), dtype=np.float32)
        categorical_feature_names: List[str] = []
        if categorical_cols:
            encoder = make_one_hot_encoder()
            cat_values = df[[column for _, column in categorical_cols]].fillna("Missing").astype(str).values
            x_categorical = encoder.fit_transform(cat_values).astype(np.float32)
            categorical_feature_names = list(encoder.get_feature_names_out([name for name, _ in categorical_cols]))

        x_all = np.concatenate([x_numeric, x_categorical], axis=1).astype(np.float32)
        feature_names = numeric_feature_names + categorical_feature_names
        return {"scaler": scaler, "encoder": encoder, "feature_names": feature_names}

    def transform(df: pd.DataFrame, transformer: Dict[str, object]) -> np.ndarray:
        x_numeric = np.empty((len(df), 0), dtype=np.float32)
        if numeric_cols:
            numeric_values = df[[column for _, column in numeric_cols]].astype(float).fillna(0.0).values
            x_numeric = transformer["scaler"].transform(numeric_values).astype(np.float32)

        x_categorical = np.empty((len(df), 0), dtype=np.float32)
        if categorical_cols:
            cat_values = df[[column for _, column in categorical_cols]].fillna("Missing").astype(str).values
            x_categorical = transformer["encoder"].transform(cat_values).astype(np.float32)

        return np.concatenate([x_numeric, x_categorical], axis=1).astype(np.float32)

    transformer = fit_transformer(train_df)
    trainval_transformer = fit_transformer(trainval_df)

    return FeatureArtifacts(
        x_train=transform(train_df, transformer),
        x_val=transform(val_df, transformer),
        x_test=transform(test_df, transformer),
        trainval_transformer=trainval_transformer,
        x_trainval=transform(trainval_df, trainval_transformer),
        x_test_final=transform(test_df, trainval_transformer),
        feature_names=transformer["feature_names"],
    )


def build_embedding_feature_matrix(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trainval_df: pd.DataFrame,
) -> FeatureArtifacts:
    embedding_cols = [column for column in train_df.columns if column.startswith("embedding_")]
    if not embedding_cols:
        raise RuntimeError("Embedding transfer requested but no embedding_* columns were found.")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[embedding_cols].astype(float).values).astype(np.float32)
    x_val = scaler.transform(val_df[embedding_cols].astype(float).values).astype(np.float32)
    x_test = scaler.transform(test_df[embedding_cols].astype(float).values).astype(np.float32)

    scaler_trainval = StandardScaler()
    x_trainval = scaler_trainval.fit_transform(trainval_df[embedding_cols].astype(float).values).astype(np.float32)
    x_test_final = scaler_trainval.transform(test_df[embedding_cols].astype(float).values).astype(np.float32)

    return FeatureArtifacts(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        trainval_transformer={"scaler": scaler_trainval},
        x_trainval=x_trainval,
        x_test_final=x_test_final,
        feature_names=embedding_cols,
    )


def get_labtrans(loss_function: str, num_durations: int):
    if loss_function == "cox_ph":
        return None
    if loss_function == "cox_time":
        return CoxTime.label_transform()
    if loss_function == "pc_hazard":
        return PCHazard.label_transform(num_durations)
    if loss_function == "logistic_hazard":
        return LogisticHazard.label_transform(num_durations)
    raise ValueError(f"Unsupported loss function: {loss_function}")


def build_survival_model(
    loss_function: str,
    input_dim: int,
    hidden_layers: Sequence[int],
    dropout: float,
    lr: float,
    labtrans,
    device: str,
    embedding_transfer: bool = False,
    embedding_mlp: bool = False,
):
    optimizer = tt.optim.Adam(lr)
    if loss_function == "cox_ph":
        if embedding_transfer and not embedding_mlp:
            net = LayerNormLinearHead(input_dim)
        else:
            net = tt.practical.MLPVanilla(
                input_dim,
                list(hidden_layers),
                1,
                batch_norm=True,
                dropout=dropout,
                output_bias=False,
            )
        model = CoxPH(net, optimizer, device=device)
    elif loss_function == "cox_time":
        net = MLPVanillaCoxTime(input_dim, list(hidden_layers), dropout=dropout)
        model = CoxTime(net, optimizer, labtrans=labtrans, device=device)
    elif loss_function == "pc_hazard":
        out_features = labtrans.out_features
        net = tt.practical.MLPVanilla(input_dim, list(hidden_layers), out_features, dropout=dropout)
        model = PCHazard(net, optimizer, duration_index=labtrans.cuts, device=device)
    elif loss_function == "logistic_hazard":
        out_features = labtrans.out_features
        net = tt.practical.MLPVanilla(input_dim, list(hidden_layers), out_features, dropout=dropout)
        model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts, device=device)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")
    return model


def get_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return df["PFS_MONTHS"].values.astype(np.float32), df["EVENT"].values.astype(np.int64)


def infer_best_epoch(log) -> int:
    history = log.to_pandas()
    val_candidates = [column for column in history.columns if "val" in column.lower() and "loss" in column.lower()]
    if not val_candidates:
        raise RuntimeError("Unable to infer validation loss column from training log.")
    val_column = val_candidates[0]
    best_index = history[val_column].astype(float).idxmin()
    return int(best_index) + 1


def compute_common_times(df: pd.DataFrame, bins: int) -> np.ndarray:
    event_times = df.loc[df["EVENT"] == 1, "PFS_MONTHS"]
    max_time = float(event_times.max()) if not event_times.empty else float(df["PFS_MONTHS"].max())
    max_time = max(max_time, 1.0)
    return np.linspace(0.0, max_time, bins)


def interpolate_survival(surv_df: pd.DataFrame, common_times: np.ndarray) -> pd.DataFrame:
    union_index = np.union1d(surv_df.index.values.astype(float), common_times)
    return (
        surv_df.reindex(union_index)
        .sort_index()
        .interpolate(method="index")
        .loc[common_times]
        .ffill()
        .bfill()
    )


def predict_risk_scores(model, x: np.ndarray) -> np.ndarray:
    risk = model.predict(x)
    risk = np.asarray(risk, dtype=float).reshape(-1)
    return risk


def derive_drug_risk_thresholds(
    val_df: pd.DataFrame,
    risk_scores: np.ndarray,
    method: str,
    min_group_fraction: float,
    min_group_size: int,
    quantile_low: float,
    quantile_high: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    val_scored = val_df[["ROW_ID", "PATIENT_ID", "SAMPLE_ID", "RSI_DRUG", "PFS_MONTHS", "EVENT", "PFS_STATUS"]].copy()
    val_scored["risk_score"] = risk_scores

    threshold_rows: List[Dict[str, object]] = []
    assignment_rows: List[Dict[str, object]] = []

    for drug, drug_df in val_scored.groupby("RSI_DRUG", sort=True):
        drug_df = drug_df.copy()
        n_rows = len(drug_df)
        effective_min_group = max(min_group_size, int(math.ceil(min_group_fraction * n_rows)))
        effective_min_group = min(effective_min_group, max(1, n_rows // 2))

        cut = float(np.median(drug_df["risk_score"].values))
        selected_method = "median"
        best_stat = float("-inf")
        candidate_count = 0

        if method == "val_logrank" and n_rows >= max(2 * effective_min_group, 4):
            q_low = float(np.quantile(drug_df["risk_score"].values, quantile_low))
            q_high = float(np.quantile(drug_df["risk_score"].values, quantile_high))
            candidate_values = sorted(
                {
                    float(value)
                    for value in drug_df["risk_score"].values
                    if q_low <= float(value) <= q_high
                }
            )

            for candidate in candidate_values:
                low_mask = drug_df["risk_score"].values <= candidate
                high_mask = ~low_mask
                if low_mask.sum() < effective_min_group or high_mask.sum() < effective_min_group:
                    continue
                if drug_df.loc[low_mask, "EVENT"].sum() == 0 and drug_df.loc[high_mask, "EVENT"].sum() == 0:
                    continue

                try:
                    result = logrank_test(
                        drug_df.loc[low_mask, "PFS_MONTHS"].values,
                        drug_df.loc[high_mask, "PFS_MONTHS"].values,
                        event_observed_A=drug_df.loc[low_mask, "EVENT"].values,
                        event_observed_B=drug_df.loc[high_mask, "EVENT"].values,
                    )
                except Exception:
                    continue

                candidate_count += 1
                test_statistic = float(result.test_statistic) if result.test_statistic is not None else float("-inf")
                if np.isnan(test_statistic):
                    continue
                if test_statistic > best_stat:
                    best_stat = test_statistic
                    cut = float(candidate)
                    selected_method = "val_logrank"

        low_mask = drug_df["risk_score"].values <= cut
        high_mask = ~low_mask
        drug_df["predicted_group"] = np.where(low_mask, "sensitive", "resistant")

        threshold_rows.append(
            {
                "RSI_DRUG": drug,
                "risk_cutoff": cut,
                "threshold_method_requested": method,
                "threshold_method_used": selected_method,
                "n_val_rows": int(n_rows),
                "n_val_sensitive": int(low_mask.sum()),
                "n_val_resistant": int(high_mask.sum()),
                "n_val_events": int(drug_df["EVENT"].sum()),
                "candidate_count": int(candidate_count),
                "min_group_size_used": int(effective_min_group),
                "quantile_low": float(quantile_low),
                "quantile_high": float(quantile_high),
                "best_logrank_statistic": None if best_stat == float("-inf") else float(best_stat),
            }
        )

        assignment_rows.extend(drug_df.to_dict("records"))

    return pd.DataFrame(threshold_rows), pd.DataFrame(assignment_rows)


def run_nested_cv(
    patient_df: pd.DataFrame,
    loss_function: str,
    extra_covariates: Sequence[str],
    include_drug_mode: str,
    include_auc: bool,
    output_risk_scores: bool,
    risk_threshold_method: str,
    risk_threshold_min_group_fraction: float,
    risk_threshold_min_group_size: int,
    risk_threshold_quantile_low: float,
    risk_threshold_quantile_high: float,
    embedding_transfer: bool,
    embedding_mlp: bool,
    hidden_layers: Sequence[int],
    transform_dropout: float,
    lr: float,
    epochs: int,
    batch_size: Optional[int],
    num_durations: int,
    outer_folds: int,
    time_bins: int,
    seed: int,
    device: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if output_risk_scores and loss_function != "cox_ph":
        raise ValueError("--risk-score is only supported with --loss-function cox_ph.")
    if embedding_transfer and loss_function != "cox_ph":
        raise ValueError("--embedding-transfer is only supported with --loss-function cox_ph.")

    unique_patients = np.array(sorted(patient_df["PATIENT_ID"].unique()))
    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    common_times = None if output_risk_scores else compute_common_times(patient_df, time_bins)

    prediction_chunks: List[pd.DataFrame] = []
    fold_summaries: List[Dict[str, object]] = []
    threshold_chunks: List[pd.DataFrame] = []
    assignment_chunks: List[pd.DataFrame] = []

    for fold_id, (trainval_idx, test_idx) in enumerate(outer_kf.split(unique_patients), start=1):
        trainval_patients = unique_patients[trainval_idx]
        test_patients = unique_patients[test_idx]

        train_patients, val_patients = train_test_split(
            trainval_patients,
            test_size=0.25,
            random_state=seed + fold_id,
            shuffle=True,
        )

        train_df = patient_df[patient_df["PATIENT_ID"].isin(train_patients)].copy()
        val_df = patient_df[patient_df["PATIENT_ID"].isin(val_patients)].copy()
        test_df = patient_df[patient_df["PATIENT_ID"].isin(test_patients)].copy()
        trainval_df = patient_df[patient_df["PATIENT_ID"].isin(trainval_patients)].copy()

        include_drug = False
        if not embedding_transfer:
            if include_drug_mode == "always":
                include_drug = True
            elif include_drug_mode == "auto":
                include_drug = trainval_df["RSI_DRUG"].nunique() > 1

        if embedding_transfer:
            features = build_embedding_feature_matrix(train_df, val_df, test_df, trainval_df)
        else:
            features = build_feature_matrix(
                train_df,
                val_df,
                test_df,
                trainval_df,
                extra_covariates,
                include_drug,
                include_auc,
            )

        labtrans = get_labtrans(loss_function, num_durations)
        if loss_function == "cox_ph":
            y_train = get_targets(train_df)
            y_val = get_targets(val_df)
        else:
            y_train = labtrans.fit_transform(*get_targets(train_df))
            y_val = labtrans.transform(*get_targets(val_df))

        if batch_size is None:
            effective_batch_size = max(1, len(train_df))
        else:
            effective_batch_size = batch_size

        model = build_survival_model(
            loss_function=loss_function,
            input_dim=features.x_train.shape[1],
            hidden_layers=hidden_layers,
            dropout=transform_dropout,
            lr=lr,
            labtrans=labtrans,
            device=device,
            embedding_transfer=embedding_transfer,
            embedding_mlp=embedding_mlp,
        )

        if loss_function == "cox_time":
            val_data = tt.tuplefy(features.x_val, y_val).repeat(10).cat()
        else:
            val_data = (features.x_val, y_val)

        best = tt.callbacks.BestWeights(metric="loss", dataset="val", load_best=True)
        log = model.fit(
            features.x_train,
            y_train,
            batch_size=effective_batch_size,
            verbose=False,
            epochs=epochs,
            val_data=val_data,
            callbacks=[best],
        )
        best_epoch = infer_best_epoch(log)

        if output_risk_scores and risk_threshold_method != "none":
            val_risk_scores = predict_risk_scores(model, features.x_val)
            fold_thresholds, fold_val_assignments = derive_drug_risk_thresholds(
                val_df=val_df,
                risk_scores=val_risk_scores,
                method=risk_threshold_method,
                min_group_fraction=risk_threshold_min_group_fraction,
                min_group_size=risk_threshold_min_group_size,
                quantile_low=risk_threshold_quantile_low,
                quantile_high=risk_threshold_quantile_high,
            )
            global_val_cutoff = float(np.median(val_risk_scores))
            fold_thresholds.insert(0, "fold", fold_id)
        else:
            fold_thresholds = None
            fold_val_assignments = None
            global_val_cutoff = None

        labtrans_final = get_labtrans(loss_function, num_durations)
        if loss_function == "cox_ph":
            y_trainval = get_targets(trainval_df)
        else:
            y_trainval = labtrans_final.fit_transform(*get_targets(trainval_df))
        final_model = build_survival_model(
            loss_function=loss_function,
            input_dim=features.x_trainval.shape[1],
            hidden_layers=hidden_layers,
            dropout=transform_dropout,
            lr=lr,
            labtrans=labtrans_final,
            device=device,
            embedding_transfer=embedding_transfer,
            embedding_mlp=embedding_mlp,
        )
        final_batch_size = max(1, len(trainval_df)) if batch_size is None else batch_size
        final_model.fit(
            features.x_trainval,
            y_trainval,
            batch_size=final_batch_size,
            verbose=False,
            epochs=best_epoch,
        )

        if output_risk_scores:
            risk_scores = predict_risk_scores(final_model, features.x_test_final)
            per_row = pd.DataFrame(
                {
                    "ROW_ID": test_df["ROW_ID"].values,
                    "fold": fold_id,
                    "risk_score": risk_scores,
                }
            )
            if fold_thresholds is not None:
                test_assignments = test_df[
                    ["ROW_ID", "PATIENT_ID", "SAMPLE_ID", "RSI_DRUG", "PFS_MONTHS", "PFS_STATUS", "EVENT"]
                ].copy()
                test_assignments["risk_score"] = risk_scores
                test_assignments = test_assignments.merge(
                    fold_thresholds[["RSI_DRUG", "risk_cutoff", "threshold_method_used"]],
                    on="RSI_DRUG",
                    how="left",
                )
                test_assignments["risk_cutoff"] = test_assignments["risk_cutoff"].fillna(global_val_cutoff)
                test_assignments["threshold_method_used"] = test_assignments["threshold_method_used"].fillna("global_val_median")
                test_assignments["predicted_group"] = np.where(
                    test_assignments["risk_score"] <= test_assignments["risk_cutoff"],
                    "sensitive",
                    "resistant",
                )
                test_assignments.insert(0, "fold", fold_id)
                assignment_chunks.append(test_assignments)
                threshold_chunks.append(fold_thresholds)
        elif loss_function in {"cox_ph", "cox_time"}:
            _ = final_model.compute_baseline_hazards()
            surv_native = final_model.predict_surv_df(features.x_test_final)
            surv_interp = interpolate_survival(surv_native, common_times)
            per_row = surv_interp.transpose().reset_index(drop=True)
            per_row.insert(0, "ROW_ID", test_df["ROW_ID"].values)
            per_row.insert(1, "fold", fold_id)
        elif loss_function == "pc_hazard":
            final_model.sub = 10
            surv_native = final_model.predict_surv_df(features.x_test_final)
            surv_interp = interpolate_survival(surv_native, common_times)
            per_row = surv_interp.transpose().reset_index(drop=True)
            per_row.insert(0, "ROW_ID", test_df["ROW_ID"].values)
            per_row.insert(1, "fold", fold_id)
        elif loss_function == "logistic_hazard":
            surv_native = final_model.interpolate(50).predict_surv_df(features.x_test_final)
            surv_interp = interpolate_survival(surv_native, common_times)
            per_row = surv_interp.transpose().reset_index(drop=True)
            per_row.insert(0, "ROW_ID", test_df["ROW_ID"].values)
            per_row.insert(1, "fold", fold_id)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        prediction_chunks.append(per_row)

        fold_summaries.append(
            {
                "fold": fold_id,
                "n_train_rows": int(len(train_df)),
                "n_val_rows": int(len(val_df)),
                "n_test_rows": int(len(test_df)),
                "n_train_patients": int(len(train_patients)),
                "n_val_patients": int(len(val_patients)),
                "n_test_patients": int(len(test_patients)),
                "n_train_samples": int(train_df["SAMPLE_ID"].nunique()),
                "n_val_samples": int(val_df["SAMPLE_ID"].nunique()),
                "n_test_samples": int(test_df["SAMPLE_ID"].nunique()),
                "best_epoch": int(best_epoch),
                "include_drug_feature": bool(include_drug),
                "risk_threshold_method": risk_threshold_method,
                "feature_names": features.feature_names,
            }
        )

    pred_df = pd.concat(prediction_chunks, ignore_index=True)
    pred_df = pred_df.sort_values("ROW_ID").reset_index(drop=True)
    if not output_risk_scores:
        time_columns = {column: f"time_{common_times[i]:.4f}" for i, column in enumerate(pred_df.columns[2:])}
        pred_df = pred_df.rename(columns=time_columns)

    fold_summary_df = pd.DataFrame(fold_summaries)
    threshold_df = pd.concat(threshold_chunks, ignore_index=True) if threshold_chunks else None
    assignment_df = pd.concat(assignment_chunks, ignore_index=True) if assignment_chunks else None
    return pred_df, fold_summary_df, threshold_df, assignment_df


def build_auc_only_output(patient_df: pd.DataFrame) -> pd.DataFrame:
    return patient_df[
        [
            "ROW_ID",
            "PATIENT_ID",
            "SAMPLE_ID",
            "RSI_DRUG",
            "predicted_auc",
            "PFS_MONTHS",
            "PFS_STATUS",
        ]
    ].copy()


def parse_args() -> argparse.Namespace:
    base = Path("/Users/idekerlab/Downloads/G2PT-DR/msk_chord_2024")
    parser = argparse.ArgumentParser(description="Generalized DRPT transfer to MSK-CHORD RSI patients.")
    parser.add_argument("--drpt-bootstrap", type=Path, required=True, help="Python module that builds DRPT objects.")
    parser.add_argument("--cell-line-model", type=str, required=True, help="Path to trained DRPT model .pt file.")
    parser.add_argument("--cell-line-response", type=str, required=True, help="Cell line response table used to build the compound encoder.")
    parser.add_argument("--cell-gene2ind", type=str, default=str(base / "gene2ind_ctg_av.txt"))
    parser.add_argument("--ontology", type=str, required=True, help="Ontology file used by MutTreeParser.")
    parser.add_argument("--patient-table", type=Path, default=base / "msk_patients_rsi_drugs.txt")
    parser.add_argument("--patient-mutations", type=Path, default=base / "cell2mutation_msk.txt")
    parser.add_argument("--patient-amplifications", type=Path, default=base / "cell2cnamplification_msk.txt")
    parser.add_argument("--patient-deletions", type=Path, default=base / "cell2cndeletion_msk.txt")
    parser.add_argument("--patient2ind", type=Path, default=base / "patient2ind_msk.txt")
    parser.add_argument("--loss-function", choices=["cox_ph", "cox_time", "pc_hazard", "logistic_hazard", "auc_only"], default="cox_ph")
    parser.add_argument("--auc-transfer", action="store_true", help="Use the current DRPT-predicted AUC transfer strategy. This is the default if no transfer mode flag is set.")
    parser.add_argument("--embedding-transfer", action="store_true", help="Use concatenated DRPT drug-system and drug-gene embeddings as the transfer features. Only supported with CoxPH.")
    parser.add_argument("--no-auc", action="store_true", help="Do not compute or use DRPT-predicted AUC as an input feature.")
    parser.add_argument("--embedding-mlp", action="store_true", help="When using --embedding-transfer with CoxPH, use a two-layer MLP instead of a LayerNorm+Linear head.")
    parser.add_argument("--risk-score", action="store_true", help="For CoxPH, output one risk score per patient sample instead of survival probabilities.")
    parser.add_argument("--drugs", nargs="*", default=None, help="Optional subset of RSI drugs to include.")
    parser.add_argument("--extra-covariates", nargs="*", default=[], choices=sorted(OPTIONAL_COVARIATES.keys()))
    parser.add_argument("--include-drug-feature", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--hidden-layers", type=int, nargs="*", default=[32, 32])
    parser.add_argument("--transform-dropout", type=float, default=0.1)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--num-durations", type=int, default=10)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden-dims", type=int, default=128, help="Hidden dimension used by the DRPT model.")
    parser.add_argument("--drpt-dropout", type=float, default=0.2, help="Dropout used when rebuilding the DRPT model.")
    parser.add_argument("--diff-transformer", dest="diff_transformer", action="store_true")
    parser.add_argument("--no-diff-transformer", dest="diff_transformer", action="store_false")
    parser.set_defaults(diff_transformer=True)
    parser.add_argument("--sys2gene", dest="sys2gene", action="store_true")
    parser.add_argument("--no-sys2gene", dest="sys2gene", action="store_false")
    parser.set_defaults(sys2gene=True)
    parser.add_argument("--gene2drug", dest="gene2drug", action="store_true")
    parser.add_argument("--no-gene2drug", dest="gene2drug", action="store_false")
    parser.set_defaults(gene2drug=True)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--auc-batch-size", type=int, default=64, help="Batch size used when predicting DRPT AUC on patient samples.")
    parser.add_argument("--risk-threshold-method", choices=["none", "median", "val_logrank"], default="none", help="When using --loss-function cox_ph --risk-score, derive drug-specific validation-set cutoffs and apply them to held-out test rows.")
    parser.add_argument("--risk-threshold-min-group-fraction", type=float, default=0.2, help="Minimum fraction of validation rows per side when searching risk cutoffs.")
    parser.add_argument("--risk-threshold-min-group-size", type=int, default=10, help="Minimum number of validation rows per side when searching risk cutoffs.")
    parser.add_argument("--risk-threshold-quantile-low", type=float, default=0.2, help="Lower quantile bound for candidate validation risk cutoffs.")
    parser.add_argument("--risk-threshold-quantile-high", type=float, default=0.8, help="Upper quantile bound for candidate validation risk cutoffs.")
    parser.add_argument("--output-prefix", type=Path, default=base / "msk_drpt_transfer")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transfer_flag_count = int(bool(args.auc_transfer)) + int(bool(args.embedding_transfer)) + int(bool(args.no_auc))
    if transfer_flag_count > 1:
        raise ValueError("Use at most one of --auc-transfer, --embedding-transfer, and --no-auc.")
    if transfer_flag_count == 0:
        args.auc_transfer = True

    if args.no_auc and args.loss_function == "auc_only":
        raise ValueError("--no-auc cannot be used together with --loss-function auc_only.")
    if args.risk_score and args.loss_function != "cox_ph":
        raise ValueError("--risk-score is only supported with --loss-function cox_ph.")
    if args.embedding_transfer and args.loss_function != "cox_ph":
        raise ValueError("--embedding-transfer is only supported with --loss-function cox_ph.")
    if args.embedding_mlp and not args.embedding_transfer:
        raise ValueError("--embedding-mlp can only be used together with --embedding-transfer.")
    if args.risk_threshold_method != "none" and not args.risk_score:
        raise ValueError("--risk-threshold-method requires --risk-score.")
    if args.risk_threshold_method != "none" and args.loss_function != "cox_ph":
        raise ValueError("--risk-threshold-method is only supported with --loss-function cox_ph.")
    if not (0.0 <= args.risk_threshold_quantile_low < args.risk_threshold_quantile_high <= 1.0):
        raise ValueError("Risk-threshold quantiles must satisfy 0 <= low < high <= 1.")

    bootstrap_module = load_bootstrap_module(args.drpt_bootstrap)
    bundle = bootstrap_module.build_drpt_components(
        cell_line_model=args.cell_line_model,
        cell_line_response=args.cell_line_response,
        cell_gene2ind=args.cell_gene2ind,
        ontology=args.ontology,
        hidden_dims=args.hidden_dims,
        dropout=args.drpt_dropout,
        diff_transformer=args.diff_transformer,
        device=device,
    )

    required_keys = {"compound_encoder", "tree_parser", "cell_response_model"}
    missing = required_keys - set(bundle)
    if missing:
        raise RuntimeError(f"Bootstrap module did not return required keys: {sorted(missing)}")

    patient_df = pd.read_csv(args.patient_table, sep="\t")
    raw_patient_count = len(patient_df)
    patient_df = normalize_patient_table(patient_df, args.drugs)
    normalized_patient_count = len(patient_df)
    if patient_df.empty:
        raise ValueError(
            "No rows remain after normalizing the patient table. "
            f"Started with {raw_patient_count} rows and ended with 0. "
            "For TCGA, check that RSI_DRUG names map to the supported drug list and that "
            "GENDER, AGE/CURRENT_AGE_DEID, PFS_MONTHS, and PFS_STATUS are populated."
        )

    sample2ind = load_patient_index(args.patient2ind)
    patient_df["LOOKUP_SAMPLE_ID"] = patient_df["SAMPLE_ID"].map(lambda sample_id: resolve_lookup_sample_id(sample_id, sample2ind))
    matched_sample_mask = patient_df["LOOKUP_SAMPLE_ID"].notna()
    matched_sample_count = int(matched_sample_mask.sum())
    unmatched_examples = (
        patient_df.loc[~matched_sample_mask, "SAMPLE_ID"].astype(str).drop_duplicates().head(10).tolist()
    )
    patient_df = patient_df[matched_sample_mask].copy().reset_index(drop=True)
    if patient_df.empty:
        raise ValueError(
            "No rows remain after matching SAMPLE_ID values to patient2ind. "
            f"Rows after normalization: {normalized_patient_count}. "
            f"Matched SAMPLE_ID rows: {matched_sample_count}. "
            f"Example unmatched SAMPLE_ID values: {unmatched_examples}"
        )
    patient_df["ROW_ID"] = np.arange(len(patient_df))

    genotype_arrays = {
        "mutation": load_binary_matrix(args.patient_mutations),
        "cna": load_binary_matrix(args.patient_amplifications),
        "cnd": load_binary_matrix(args.patient_deletions),
    }

    if args.embedding_transfer:
        patient_df["predicted_auc"] = predict_auc_for_all_rows(
            response_df=patient_df,
            sample2ind=sample2ind,
            genotype_arrays=genotype_arrays,
            compound_encoder=bundle["compound_encoder"],
            tree_parser=bundle["tree_parser"],
            cell_response_model=bundle["cell_response_model"],
            device=device,
            jobs=args.jobs,
            auc_batch_size=args.auc_batch_size,
            sys2gene=args.sys2gene,
            gene2drug=args.gene2drug,
        ).values
        embedding_df = predict_embedding_features_for_all_rows(
            response_df=patient_df,
            sample2ind=sample2ind,
            genotype_arrays=genotype_arrays,
            compound_encoder=bundle["compound_encoder"],
            tree_parser=bundle["tree_parser"],
            cell_response_model=bundle["cell_response_model"],
            device=device,
            jobs=args.jobs,
            auc_batch_size=args.auc_batch_size,
            sys2gene=args.sys2gene,
        )
        patient_df = pd.concat([patient_df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    elif not args.no_auc:
        patient_df["predicted_auc"] = predict_auc_for_all_rows(
            response_df=patient_df,
            sample2ind=sample2ind,
            genotype_arrays=genotype_arrays,
            compound_encoder=bundle["compound_encoder"],
            tree_parser=bundle["tree_parser"],
            cell_response_model=bundle["cell_response_model"],
            device=device,
            jobs=args.jobs,
            auc_batch_size=args.auc_batch_size,
            sys2gene=args.sys2gene,
            gene2drug=args.gene2drug,
        ).values

    output_prefix = args.output_prefix
    metadata_columns = [
        "ROW_ID",
        "PATIENT_ID",
        "SAMPLE_ID",
        "RSI_DRUG",
        "PFS_MONTHS",
        "PFS_STATUS",
        "EVENT",
    ]
    if "predicted_auc" in patient_df.columns:
        metadata_columns.insert(4, "predicted_auc")
    metadata_df = patient_df[metadata_columns].copy()

    if args.loss_function == "auc_only":
        auc_output = build_auc_only_output(patient_df)
        auc_path = output_prefix.with_name(output_prefix.name + "_auc_only.tsv")
        auc_output.to_csv(auc_path, sep="\t", index=False)
        config_path = output_prefix.with_name(output_prefix.name + "_config.json")
        config_path.write_text(json.dumps(vars(args), default=str, indent=2))
        print(f"Wrote AUC-only output to {auc_path}")
        print(f"Rows: {len(auc_output)}")
        return

    pred_df, fold_summary_df, threshold_df, assignment_df = run_nested_cv(
        patient_df=patient_df,
        loss_function=args.loss_function,
        extra_covariates=args.extra_covariates,
        include_drug_mode=args.include_drug_feature,
        include_auc=args.auc_transfer,
        output_risk_scores=args.risk_score,
        risk_threshold_method=args.risk_threshold_method,
        risk_threshold_min_group_fraction=args.risk_threshold_min_group_fraction,
        risk_threshold_min_group_size=args.risk_threshold_min_group_size,
        risk_threshold_quantile_low=args.risk_threshold_quantile_low,
        risk_threshold_quantile_high=args.risk_threshold_quantile_high,
        embedding_transfer=args.embedding_transfer,
        embedding_mlp=args.embedding_mlp,
        hidden_layers=args.hidden_layers,
        transform_dropout=args.transform_dropout,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_durations=args.num_durations,
        outer_folds=args.outer_folds,
        time_bins=args.time_bins,
        seed=args.seed,
        device=device,
    )

    final_df = metadata_df.merge(pred_df, on="ROW_ID", how="inner").sort_values("ROW_ID").reset_index(drop=True)

    survival_path = output_prefix.with_name(output_prefix.name + ("_risk_scores.tsv" if args.risk_score else "_survival_probabilities.tsv"))
    fold_path = output_prefix.with_name(output_prefix.name + "_fold_summary.tsv")
    config_path = output_prefix.with_name(output_prefix.name + "_config.json")
    threshold_path = output_prefix.with_name(output_prefix.name + "_fold_drug_risk_thresholds.tsv")
    assignment_path = output_prefix.with_name(output_prefix.name + "_test_risk_assignments.tsv")

    final_df.to_csv(survival_path, sep="\t", index=False)
    fold_summary_df.to_csv(fold_path, sep="\t", index=False)
    if threshold_df is not None and assignment_df is not None:
        threshold_df.to_csv(threshold_path, sep="\t", index=False)
        assignment_df.to_csv(assignment_path, sep="\t", index=False)
    config_path.write_text(json.dumps(vars(args), default=str, indent=2))

    if args.risk_score:
        print(f"Wrote risk scores to {survival_path}")
    else:
        print(f"Wrote survival probabilities to {survival_path}")
    print(f"Wrote fold summary to {fold_path}")
    if threshold_df is not None and assignment_df is not None:
        print(f"Wrote fold drug risk thresholds to {threshold_path}")
        print(f"Wrote test risk assignments to {assignment_path}")
    print(f"Rows: {len(final_df)}")


if __name__ == "__main__":
    main()
