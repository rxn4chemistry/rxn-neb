"""Core CLI for NEB retro."""
import json
from pathlib import Path
from typing import Optional

import click
from loguru import logger

from ..retro import retro_synthesis


@click.command()
@click.option("--product", required=True, type=str, default="NS(=O)(=O)c1nn(-c2ccccn2)cc1Br")
@click.option("--forward_model_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--backward_model_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--fingerprints_model_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--pca_model_filename", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--tree_data_dict_pca_filename", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--max_steps", required=False, type=int, default=15)
@click.option("--pruning_steps", required=False, type=int, default=3)
@click.option("--output_path", required=False, type=click.Path(path_type=Path), default=None)
def main(
    product: str,
    forward_model_path: Path,
    backward_model_path: Path,
    fingerprints_model_path: Path,
    pca_model_filename: Path,
    tree_data_dict_pca_filename: Path,
    max_steps: int,
    pruning_steps: int,
    output_path: Optional[Path],
) -> None:
    """Execute NEB retro.

    Args:
        product: product SMILES.
        forward_model_path: path to forward model.
        backward_model_path: path to backward model.
        fingerprints_model_path: path to fingerprints model.
        pca_model_filename: PCA model file.
        tree_data_dict_pca_filename: PCA model file fro tree data.
        max_steps: maximum number of steps for the retro.
        pruning_steps: number of steps in-between pruning.
        output_path: file where to store the retro. Defaults to None, a.k.a.,
            simply print the retro without saving it.

    Returns:
        a retrosynthesis object.
    """
    data = {
        "product": product,
        "parameters": {"max_steps": max_steps, "pruning_steps": pruning_steps},
        "forward_model_path": str(forward_model_path),
        "backward_model_path": str(backward_model_path),
        "fingerprints_model_path": str(fingerprints_model_path),
        "pca_model_filename": str(pca_model_filename),
        "tree_data_dict_pca_filename": str(tree_data_dict_pca_filename),
    }
    try:
        retrosynthesis = retro_synthesis(data)
    except Exception:
        logger.exception("NEB-retro failed")
        retrosynthesis = {"status": "DONE", "routes": []}

    if output_path:
        with open(output_path, "w") as fp:
            json.dump(retrosynthesis, fp, indent=4)
    logger.info(f"retrosynthesis={retrosynthesis}")
