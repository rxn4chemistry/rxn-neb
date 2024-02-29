#!/usr/bin/env python

from functools import lru_cache
from typing import List, Set

from loguru import logger
from rxn.chemutils.tokenization import detokenize_smiles, tokenize_smiles
from rxn.onmt_utils import Translator

from .utils.smiles import (
    from_complex_smile_to_fragments,
    multistep_standardize,
    multistep_standardize_fw,
)


@lru_cache(maxsize=5)
def load_single_step_prediction_model(model_path: str, **model_parameters) -> Translator:
    """Load single step prediction model.

    Args:
        model_path: path to the model.

    Returns:
        loaded model.
    """
    return Translator.from_model_path(model_path=model_path, **model_parameters)


def run_rxn_prediction(product: str, fw_model: Translator, re_model: Translator, verbose: bool = False) -> List[str]:
    """Run RXN roundtrip predictions.

    Args:
        product: product SMILES.
        fw_model: forward model.
        re_model: backward model.
        verbose: verbose mode. Defaults to False.

    Returns:
        list of predictions passing the roundtrip check.
    """
    prod_std = multistep_standardize(product)
    output_tokenize_smile = tokenize_smiles(prod_std)
    retro_list = [
        [detokenize_smiles(result.text) for result in result_list] for result_list in re_model.translate_multiple_with_scores([output_tokenize_smile])
    ]
    if verbose:
        for retro_smiles in retro_list[0]:
            print("retro_prediction:", retro_smiles)
    if len(retro_list) == 0:
        return []

    # round-trip
    back_pred_std = []
    for smile in retro_list[0]:
        smile = multistep_standardize_fw(smile)
        input_tokenize_smile = tokenize_smiles(smile)
        output_tokenize_smile = fw_model.translate_single(input_tokenize_smile)
        out_smile = detokenize_smiles(output_tokenize_smile)
        if out_smile in [product, prod_std]:
            back_pred_std.append(smile)
            if verbose:
                print("OK", smile, out_smile)
        else:
            if verbose:
                print("FAIL", smile, out_smile)
            pass
    if len(back_pred_std) == 0:
        return []

    # remove retro prediction if one of the precursor is the target product
    retro_predictions_: Set[str] = set()
    for _, r in enumerate(back_pred_std):
        can_fragments_, s_ = from_complex_smile_to_fragments(r)
        if verbose:
            print("CHECK:", r, "<>", s_, can_fragments_)
        if prod_std not in can_fragments_:
            retro_predictions_.add(s_)
        else:
            string_log = str(product) + " " + str(can_fragments_)
            logger.info("run_rxn_prediction: RETRO_PREDICTION_DISCARDED_product_in_fragments " + string_log)
    good_retro_predictions = list(retro_predictions_)

    return good_retro_predictions
