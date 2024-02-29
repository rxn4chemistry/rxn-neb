"""SMILES utilities."""
from typing import Set, Tuple

import regex as re
from rxn.chemutils.multicomponent_smiles import multicomponent_smiles_to_list
from rxn.chemutils.smiles_standardization import standardize_molecules, standardize_smiles

from ..availability import is_available


def multistep_standardize(smiles):
    # retro - model
    return standardize_smiles(
        smiles,
        canonicalize=True,
        sanitize=True,
        inchify=False,
    )


def multistep_standardize_fw(molecules):
    # forward - model
    return standardize_molecules(
        molecules,
        canonicalize=True,
        sanitize=True,
        inchify=False,
        fragment_bond="~",
        ordered_precursors=True,
        molecule_token_delimiter=None,
        is_enzymatic=False,
        enzyme_separator="|",
    )


def canonicalize_smiles(smile, model_selection="forward_model"):
    if model_selection == "retro_model":
        sm = multistep_standardize(smile)
    elif model_selection == "forward_model":
        sm = multistep_standardize_fw(smile)
    else:
        sm = "None"
    return sm


def try_further_splitter(s):
    s_ = s
    if re.search(r"(\[[A-Z][a-z]?\+?\])\)", s):
        s_ = re.sub(r"(\[[A-Z][a-z]?\+?\])\)", r").\1.", s)
    else:
        s_ = re.sub(r"(\[[A-Z][a-z]?\+?\])", r".\1.", s)
    s_ = re.sub(r"\.+", ".", s_)
    return s_


def try_further(s):
    outcome = False
    # repeated twice
    s = try_further_splitter(s)
    s = try_further_splitter(s)
    more_split = s.split(".")
    for ms in more_split:
        if is_available(ms):
            outcome = True
        else:
            return False
    return outcome


def simpler_smile_tilde(smile):
    # O=C1[C@@H](N~Cl~Cl)CCN1Cc1ccc2[nH]cnc2c1 => O=C1[C@@H](N)CCN1Cc1ccc2[nH]cnc2c1.Cl.Cl
    smiles = smile.split(".")
    new_smile = ""
    for smile in smiles:
        z = re.search(r"\(([A-Z][a-z]?)((?:~[A-Z][a-z]?)+)\)", smile)
        if z:
            end1 = z.span(1)[1]
            ini2 = z.span(2)[0]
            end2 = z.span(2)[1]
            t1 = re.sub(r"~", ".", smile[ini2:end2])
            smile = smile[:end1] + smile[end2:] + t1
        if len(new_smile) > 0:
            new_smile = new_smile + "." + smile
        else:
            new_smile = smile
    return new_smile


def from_complex_smile_to_fragments(multi) -> Tuple[Set[str], str]:
    can_fragments = set()
    # Clean for '(N~Cl~Cl)'
    multi = simpler_smile_tilde(multi)
    elements = multicomponent_smiles_to_list(multi, fragment_bond="~")
    for element in elements:
        fragments = element.split(".")
        for f in fragments:
            try:
                can_fragments.add(canonicalize_smiles(f))
            except Exception:
                can_fragments.add(f + "__NONE__FAILED_canonicalize_smiles")
    s_ = ""
    for e in sorted(list(can_fragments)):
        s_ = s_ + str(e) + "."
    if len(s_) > 1:
        s_ = s_[0:-1]
    return can_fragments, s_
