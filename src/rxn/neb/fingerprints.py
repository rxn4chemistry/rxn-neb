#!/usr/bin/env python

from functools import lru_cache
from pathlib import Path

from rxn.utilities.modeling.core import RXNFPModel
from rxn.utilities.modeling.tokenization import BasicSmilesTokenizer, SmilesTokenizer


@lru_cache(maxsize=5)
def load_fingerprints_model(model_path: str) -> RXNFPModel:
    """Load fingerprints model.

    Args:
        model_path: path to the model.

    Returns:
        loaded model.
    """
    return RXNFPModel(
        model_name_or_path=str(model_path),
        tokenizer=SmilesTokenizer(
            vocab_file=str(Path(model_path) / "vocab.txt"),
            basic_tokenizer=BasicSmilesTokenizer(),
        ),
    )


def compute_rxnfp(smiles: str, fingerprints_model: RXNFPModel):
    return list(fingerprints_model.predict([smiles]))[0]
