#!/usr/bin/env python3

from __future__ import annotations

import pandas as pd
import torch

from src.model.compound import DrugEmbeddingCompoundModel
from src.model.model.drug_response_model import DrugResponseModel
from src.utils.data import CompoundEncoder
from src.utils.tree import MutTreeParser


def build_drpt_components(
    cell_line_model: str,
    cell_line_response: str,
    cell_gene2ind: str,
    ontology: str,
    hidden_dims: int,
    dropout: float,
    diff_transformer: bool,
    device: str,
):
    cell_line_response_df = pd.read_csv(cell_line_response, header=None, sep="\t")

    compound_encoder = CompoundEncoder("Embedding", dataset=cell_line_response_df, out="dummy/")
    compound_model = DrugEmbeddingCompoundModel(compound_encoder.num_drugs(), hidden_dims)
    tree_parser = MutTreeParser(ontology, cell_gene2ind)

    cell_response_model = DrugResponseModel(
        tree_parser,
        ["mutation", "cna", "cnd"],
        hidden_dims,
        compound_model,
        dropout=dropout,
        diff_transformer=diff_transformer,
    )

    state = torch.load(cell_line_model, map_location=torch.device(device))
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    cell_response_model.load_state_dict(state_dict)
    cell_response_model.to(device)
    cell_response_model.eval()

    for param in cell_response_model.parameters():
        param.requires_grad = False

    return {
        "compound_encoder": compound_encoder,
        "tree_parser": tree_parser,
        "cell_response_model": cell_response_model,
    }
