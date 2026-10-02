#!/usr/bin/env python
"""Extract predictions, attention weights, and embeddings from a saved DRPT model.

Run this script from the ``g2pt_env`` Conda environment. By default, it loads
``rsi_model.pt``, performs inference on the complete RSI training dataset, and
writes ``interpretation/model_results.pkl``.
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.model.compound import DrugEmbeddingCompoundModel
from src.model.model.drug_response_model import DrugResponseModel
from src.utils.data import CompoundEncoder, move_to
from src.utils.data.dataset import DrugResponseCollator, DrugResponseDataset
from src.utils.tree import MutTreeParser


REPOSITORY_ROOT = Path(__file__).resolve().parent

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


def _read_response_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, header=None, sep="\t")
    if table.shape[1] < 3:
        raise ValueError(
            "The response table must contain at least cell line, SMILES, and response columns: "
            "{}".format(path)
        )
    return table


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def _load_state_dict(checkpoint_path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, Mapping):
        state_dict = checkpoint
    else:
        raise TypeError("Expected the model checkpoint to contain a state dictionary")

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def instantiate_model(
    response_table: pd.DataFrame,
    checkpoint_path: Path,
    ontology_path: Path,
    gene_index_path: Path,
    genotype_paths: Mapping[str, Path],
    device: torch.device,
    hidden_dims: int = 128,
    dropout: float = 0.2,
    diff_transformer: bool = True,
) -> DrugResponseModel:
    compound_encoder = CompoundEncoder("Embedding", dataset=response_table, out=None)
    compound_model = DrugEmbeddingCompoundModel(compound_encoder.num_drugs(), hidden_dims)
    tree_parser = MutTreeParser(str(ontology_path), str(gene_index_path))

    model = DrugResponseModel(
        tree_parser,
        list(genotype_paths.keys()),
        hidden_dims,
        compound_model,
        dropout=dropout,
        diff_transformer=diff_transformer,
    )
    model.load_state_dict(_load_state_dict(checkpoint_path, device))
    model.to(device)
    model.eval()
    return model


def _forward_with_intermediates(
    model: DrugResponseModel,
    genotype_dict,
    compound,
    nested_forward,
    nested_backward,
    gene_to_system_mask,
    system_to_gene_mask,
    sys2cell: bool = True,
    cell2sys: bool = True,
    sys2gene: bool = True,
    gene2drug: bool = True,
    mut2gene: bool = True,
    with_indices: bool = True,
):
    batch_size = compound.size(0)

    if mut2gene:
        gene_embedding = model.get_mut2gene(
            genotype_dict,
            with_indices=with_indices,
            batch_size=batch_size,
        )
        system_embedding = model.system_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )[:, :-1, :]
        system_embedding, gene_effect = model.get_gene2sys(
            system_embedding, gene_embedding, gene_to_system_mask
        )
        system_embedding = system_embedding + model.effect_norm(gene_effect)
    else:
        system_embedding = model.get_mut2system(
            genotype_dict,
            with_indices=with_indices,
            batch_size=batch_size,
        )
        gene_embedding = model.gene_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )[:, :-1, :]

    if sys2cell:
        system_embedding = model.get_sys2sys(
            system_embedding,
            nested_forward,
            direction="forward",
            return_updates=False,
            with_indices=False,
        )
    system_embedding_forward = system_embedding.clone()

    if cell2sys:
        system_embedding = model.get_sys2sys(
            system_embedding,
            nested_backward,
            direction="backward",
            return_updates=False,
            with_indices=False,
        )

    if sys2gene:
        gene_embedding, system_effect_on_gene = model.get_sys2gene(
            gene_embedding, system_embedding, system_to_gene_mask
        )
        gene_embedding = gene_embedding + model.effect_norm(system_effect_on_gene)

    compound_embedding = model.get_compound_embedding(compound, unsqueeze=True)
    if gene2drug:
        prediction = model.prediction(
            model.predictor_genes_systems,
            compound_embedding,
            system_embedding,
            gene_embedding,
        )
    else:
        prediction = model.prediction(
            model.predictor_systems,
            compound_embedding,
            system_embedding,
        )

    _, system_attention = model.get_system2comp(
        compound_embedding, system_embedding, attention=True, score=False
    )
    _, gene_attention = model.get_gene2comp(
        compound_embedding, gene_embedding, attention=True, score=False
    )

    return (
        prediction,
        system_attention,
        gene_attention,
        system_embedding,
        gene_embedding,
        system_embedding_forward,
    )


def extract_model_results(
    model: DrugResponseModel,
    response_table: pd.DataFrame,
    cell_index_path: Path,
    genotype_paths: Mapping[str, Path],
    ontology_path: Path,
    gene_index_path: Path,
    device: torch.device,
    smiles_to_name: Mapping[str, str],
    batch_size: int = 32,
    num_workers: int = 0,
    sys2gene: bool = True,
    gene2drug: bool = True,
) -> Dict[str, Dict[str, object]]:
    compound_encoder = CompoundEncoder("Embedding", dataset=response_table, out=None)
    tree_parser = MutTreeParser(str(ontology_path), str(gene_index_path))
    genotype_strings = {name: str(path) for name, path in genotype_paths.items()}

    dataset = DrugResponseDataset(
        response_table,
        str(cell_index_path),
        genotype_strings,
        compound_encoder,
        tree_parser,
        mut2gene=True,
        with_indices=True,
    )
    collator = DrugResponseCollator(
        tree_parser,
        list(genotype_paths.keys()),
        compound_encoder,
        mut2gene=True,
        with_indices=True,
    )
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    nested_forward = move_to(
        tree_parser.get_nested_subtree_mask(["default"], direction="forward"), device
    )
    nested_backward = move_to(
        tree_parser.get_nested_subtree_mask(["default"], direction="backward"), device
    )
    gene_to_system_mask = move_to(
        torch.tensor(tree_parser.gene2sys_mask, dtype=torch.bool), device
    )
    system_to_gene_mask = move_to(
        torch.tensor(tree_parser.sys2gene_mask, dtype=torch.bool), device
    )

    collected = {
        "predictions": [],
        "system_attn": [],
        "gene_attn": [],
        "system_embeddings": [],
        "gene_embeddings": [],
        "system_embeddings_forward": [],
    }

    model.eval()
    with torch.no_grad():
        for batch_number, batch in enumerate(dataloader, start=1):
            batch = move_to(batch, device)
            outputs = _forward_with_intermediates(
                model,
                batch["genotype"],
                batch["drug"],
                nested_forward,
                nested_backward,
                gene_to_system_mask,
                system_to_gene_mask,
                sys2cell=True,
                cell2sys=True,
                sys2gene=sys2gene,
                gene2drug=gene2drug,
                mut2gene=True,
                with_indices=True,
            )
            prediction, system_attn, gene_attn, system_emb, gene_emb, system_forward = outputs
            current_batch_size = prediction.shape[0]

            collected["predictions"].append(
                prediction.detach().cpu().numpy().reshape(current_batch_size, -1)[:, 0]
            )
            collected["system_attn"].append(
                system_attn.detach().cpu().numpy().reshape(current_batch_size, -1)
            )
            collected["gene_attn"].append(
                gene_attn.detach().cpu().numpy().reshape(current_batch_size, -1)
            )
            collected["system_embeddings"].append(system_emb.detach().cpu().numpy())
            collected["gene_embeddings"].append(gene_emb.detach().cpu().numpy())
            collected["system_embeddings_forward"].append(
                system_forward.detach().cpu().numpy()
            )

            if batch_number % 25 == 0 or batch_number == len(dataloader):
                print("Processed batch {}/{}".format(batch_number, len(dataloader)))

    arrays = {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}
    results = {}
    grouped_indices = response_table.groupby(1, sort=False).indices

    missing_smiles = sorted(str(smiles) for smiles in grouped_indices if smiles not in smiles_to_name)
    if missing_smiles:
        raise KeyError(
            "No drug-name mapping is available for these SMILES strings: {}".format(
                missing_smiles
            )
        )

    for smiles, row_indices in grouped_indices.items():
        indices = np.asarray(row_indices, dtype=int)
        drug_name = smiles_to_name[smiles]
        results[drug_name] = {
            "celllines": response_table.iloc[indices, 0].astype(str).tolist(),
            "actual": response_table.iloc[indices, 2].tolist(),
            "predictions": arrays["predictions"][indices],
            "system_attn": arrays["system_attn"][indices],
            "gene_attn": arrays["gene_attn"][indices],
            "system_embeddings": arrays["system_embeddings"][indices],
            "gene_embeddings": arrays["gene_embeddings"][indices],
            "system_embeddings_forward": arrays["system_embeddings_forward"][indices],
        }

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPOSITORY_ROOT / "rsi_model.pt")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "data/full_model/train_dataset_rsi.txt",
        help="Response table used both to reconstruct the drug encoder and for inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "interpretation/model_results.pkl",
    )
    parser.add_argument(
        "--ontology", type=Path, default=REPOSITORY_ROOT / "data/ontology_ctg_av.txt"
    )
    parser.add_argument(
        "--gene-index", type=Path, default=REPOSITORY_ROOT / "data/gene2ind_ctg_av.txt"
    )
    parser.add_argument(
        "--cell-index", type=Path, default=REPOSITORY_ROOT / "data/cell2ind_av.txt"
    )
    parser.add_argument(
        "--mutation", type=Path, default=REPOSITORY_ROOT / "data/cell2mutation_ctg_av.txt"
    )
    parser.add_argument(
        "--amplification",
        type=Path,
        default=REPOSITORY_ROOT / "data/cell2cnamplification_ctg_av.txt",
    )
    parser.add_argument(
        "--deletion",
        type=Path,
        default=REPOSITORY_ROOT / "data/cell2cndeletion_ctg_av.txt",
    )
    parser.add_argument("--hidden-dims", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-diff-transformer", action="store_true")
    parser.add_argument("--no-sys2gene", action="store_true")
    parser.add_argument("--no-gene2drug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    required_paths: Sequence[Path] = (
        args.model,
        args.dataset,
        args.ontology,
        args.gene_index,
        args.cell_index,
        args.mutation,
        args.amplification,
        args.deletion,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files: {}".format(missing))
    if args.output.exists() and not args.force:
        raise FileExistsError(
            "Output already exists: {}. Pass --force to replace it.".format(args.output)
        )

    device = _resolve_device(args.device)
    response_table = _read_response_table(args.dataset)
    genotype_paths = {
        "mutation": args.mutation,
        "cna": args.amplification,
        "cnd": args.deletion,
    }

    print("Loading model from {}".format(args.model))
    print("Running inference for {} rows on {}".format(len(response_table), device))
    model = instantiate_model(
        response_table=response_table,
        checkpoint_path=args.model,
        ontology_path=args.ontology,
        gene_index_path=args.gene_index,
        genotype_paths=genotype_paths,
        device=device,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        diff_transformer=not args.no_diff_transformer,
    )
    results = extract_model_results(
        model=model,
        response_table=response_table,
        cell_index_path=args.cell_index,
        genotype_paths=genotype_paths,
        ontology_path=args.ontology,
        gene_index_path=args.gene_index,
        device=device,
        smiles_to_name=SMILES_TO_NAME,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sys2gene=not args.no_sys2gene,
        gene2drug=not args.no_gene2drug,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    try:
        with temporary_output.open("wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(str(temporary_output), str(args.output))
    finally:
        if temporary_output.exists():
            temporary_output.unlink()

    print("Saved {} drug result dictionaries to {}".format(len(results), args.output))


if __name__ == "__main__":
    main()
