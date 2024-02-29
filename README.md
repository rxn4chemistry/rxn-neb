# rxn-neb

## Setup

To install the package run:

```console
poetry install
```

## Pre-process data for running rxn-neb

Here we assume to start from a JSON file reporting synthesis trees characterized by reactions SMILES represented using a pre-order traversal. A sample is provided [here](./sample-data/reaction_trees.json).

Additionally we assume a model for reaction fingerprints compatible with [`rxnfp`](https://github.com/rxn4chemistry/rxnfp) is available (see the repo for instructions on how to train your own on public or proprietary data).

To get the default model used in RXN for Chemistry simply clone the repo:

```console
git clone https://github.com/rxn4chemistry/rxnfp.git
```

You can directly use the default model available at `./rxnfp/rxnfp/models/transformers/bert_ft`.

Prepare the fingerprints from available synthesis trees:

```console
generate-fingerprints --reaction_trees_path "./sample-data/reaction_trees.json" --fingerprints_model_path "./rxnfp/rxnfp/models/transformers/bert_ft" --generated_fingerprints_path "./sandbox/generated_fingerprints.csv"
```

Prepare the PCA model for fingerprint compression and related indexes:

```console
generate-pca-compression-and-indices  --reaction_trees_path "./sample-data/reaction_trees.json" --fingerprints_path "./sandbox/generated_fingerprints.csv" --pca_model_filename "./sandbox/pca.pkl" --tree_data_dict_pca_filename "./sandbox/tree_data_dict_pca.pkl"
```

NOTE: these examples are creating a `sandbox` folder where all outputs are stored.

## Usage

We assume you have a pair of single-step forward and backward model trained using [`rxn-onmt-models`](https://github.com/rxn4chemistry/rxn-onmt-models) (see the repo for a detailed guide on how to train them on public or proprietary data).

```console
run-neb-retrosynthesis --product "NS(=O)(=O)c1nn(-c2ccccn2)cc1Br" \
    --forward_model_path "/path/to/forward_model.pt" \
    --backward_model_path "/path/to/backward_model.pt" \
    --fingerprints_model_path "./rxnfp/rxnfp/models/transformers/bert_ft"  \
    --pca_model_filename "./sandbox/pca.pkl" \
    --tree_data_dict_pca_filename "./sandbox/tree_data_dict_pca.pkl" \
    --output_path ./test_retro.json
```
