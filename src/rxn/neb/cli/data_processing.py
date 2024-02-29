"""Data preparation CLI entry-points."""
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import click
import numpy as np
import pandas as pd
from rxn.neb.fingerprints import load_fingerprints_model
from rxn.neb.utils.general import batcher
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


@click.command()
@click.option("--reaction_trees_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--fingerprints_model_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--generated_fingerprints_path", required=True, type=click.Path(path_type=Path))
@click.option("--batch_size", required=False, type=int, default=1024)
def generate_fingerprints(
    reaction_trees_path: Path,
    fingerprints_model_path: Path,
    generated_fingerprints_path: Path,
    batch_size: int,
) -> None:
    """Generate fingerprints starting from reaction trees in .JSON format.

    Args:
        reaction_trees_path: path to .JSON for the reaction trees.
        fingerprints_model_path: path to fingerprints model.
        generated_fingerprints_path: path to generated fingerprints.
        batch_size: size of the batch used for dumping fingerprints to file. Defaults to 1024.
    """
    with open(reaction_trees_path) as fp:
        trees: Dict[str, Dict[str, Any]] = json.load(fp)

    fingerprint_model = load_fingerprints_model(fingerprints_model_path)

    # NOTE: sorted reactions according to rxnfp_ids from reaction trees
    rxns = [
        rxn
        for _, rxn in sorted(
            ({index: rxn for tree in trees.values() for index, rxn in zip(tree["rxnfp_ids"], tree["rxns"])}).items(),
            key=lambda index_to_rxn: index_to_rxn[0],
        )
    ]

    if generated_fingerprints_path.exists():
        generated_fingerprints_path.unlink()
    os.makedirs(generated_fingerprints_path.parent, exist_ok=True)
    for fingerprints_batch in batcher(fingerprint_model.predict(rxns), batch_size=batch_size):
        pd.DataFrame(fingerprints_batch).to_csv(generated_fingerprints_path, mode="ab", header=False, index=None)


@click.command()
@click.option("--reaction_trees_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--fingerprints_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--pca_model_filename", required=True, type=click.Path(path_type=Path))
@click.option("--tree_data_dict_pca_filename", required=True, type=click.Path(path_type=Path))
@click.option("--number_of_components", required=False, type=int, default=16)
@click.option("--max_depth", required=False, type=int, default=16)
@click.option("--leaf_size", required=False, type=int, default=8)
def generate_pca_compression_and_indices(
    reaction_trees_path: Path,
    fingerprints_path: Path,
    pca_model_filename: Path,
    tree_data_dict_pca_filename: Path,
    number_of_components: int,
    max_depth: int,
    leaf_size: int,
) -> None:
    """Generate a PCA compressed representation for fingerprints a set of KDTree indices.

    Args:
        reaction_trees_path: path to .JSON for the reaction trees.
        fingerprints_path: path to fingerprints.
        pca_model_filename: path to which the PCA object will be serialized to.
        tree_data_dict_pca_filename: path to which the depth to KDTree mapping object will be serialized to.
        number_of_components: number of principal components.
        max_depth: maximum depth considered.
        leaf_size: size of the KDTree leaves.
    """
    with open(reaction_trees_path) as fp:
        trees: Dict[str, Dict[str, Any]] = json.load(fp)

    fingerprints = pd.read_csv(fingerprints_path, header=None, index_col=None)

    pca = PCA(n_components=number_of_components, svd_solver="full")
    pca.fit(fingerprints.values)

    with open(pca_model_filename, "wb") as fp:
        pickle.dump(pca, fp, protocol=4)

    fingerprints_principal_components = pca.transform(fingerprints.values)
    depth_to_kdtree: Dict[str, KDTree] = {}
    for depth in range(1, max_depth):
        fingerprints_principal_components_for_depth = []
        for tree in trees.values():
            reactions = tree["rxnfp_ids"]
            for offset in range(len(reactions) - depth + 1):
                fingerprints_principal_components_for_depth.append(
                    np.array([values for index in reactions[offset : offset + depth] for values in fingerprints_principal_components[index].tolist()])
                )
        if fingerprints_principal_components_for_depth:
            kdtree = KDTree(
                np.array(fingerprints_principal_components_for_depth).reshape(
                    len(fingerprints_principal_components_for_depth), number_of_components * depth
                ),
                leaf_size=leaf_size,
            )
            depth_to_kdtree[depth] = kdtree
    with open(tree_data_dict_pca_filename, "wb") as fp:
        pickle.dump(depth_to_kdtree, fp, protocol=4)
