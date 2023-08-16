
import functools
import typing as ty

import pandas as pd
import RNA


@functools.lru_cache()
def duplex_energy(s1: str, s2: str) -> float:
    return RNA.duplexfold(s1, s2).energy


@functools.lru_cache()
def folded_sequence(sequence, model_details):
    folder = RNA.fold_compound(sequence, model_details)
    dot_bracket, mfe = folder.mfe()
    return dot_bracket, mfe


def fold_sequences(
    sequences: ty.Iterable[str], temperature: float = 37.0,
) -> pd.DataFrame:

    md = RNA.md()
    md.temperature = temperature

    seq2structure_map = {
        "sequence": [],
        f"structure_{int(temperature)}": [],
        f"mfe_{int(temperature)}": [],
    }

    for sequence in sequences:
        dot_bracket, mfe = folded_sequence(sequence, md)
        seq2structure_map["sequence"].append(sequence)
        seq2structure_map[f"structure_{int(temperature)}"].append(dot_bracket)
        seq2structure_map[f"mfe_{int(temperature)}"].append(mfe)

    return pd.DataFrame(seq2structure_map).set_index("sequence")

def fraction(seq: str, nucleoids: str) -> float:
    """Computes the fraction of the sequence string that is the set of nucleoids
    given.

    Parameters
    ----------
    seq : str
        The sequence string
    nucleoids : str
        The list of nucleoids to compute the fraction for.

    Returns
    -------
    float
        The fraction
    """
    return sum([seq.count(n) for n in nucleoids]) / len(seq)