"""Compound availabilities."""
from rxn.availability import IsAvailable
from rxn.chemutils.smiles_standardization import standardize_smiles


def standardize(smiles: str) -> str:
    """Standardize a SMILES string.

    Args:
        smiles: SMILES string.

    Returns:
        standardized SMILES.
    """
    return standardize_smiles(
        smiles,
        canonicalize=True,
        sanitize=True,
        inchify=False,
    )


is_available_object = IsAvailable(
    pricing_threshold=0.0,
    always_available=None,
    model_available=None,
    excluded=None,
    avoid_substructure=None,
    are_materials_exclusive=False,
    standardization_function=standardize,
    # NOTE: here you can include additional compounds
    # that are considered available passing a file containing
    # one smiles per line
    additional_compounds_filepath=None,
)


def is_available(smiles: str) -> bool:
    """Check whether compound is available.

    Args:
        smiles: SMILES string.

    Returns:
        whether the compound is available.
    """
    return is_available_object(smiles)
