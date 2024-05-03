import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, List

import anndata
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from anndata import AnnData
from Bio.SeqIO.FastaIO import SimpleFastaParser

logger = logging.getLogger(__name__)


def create_dirs(paths:List):
    for path in paths:
        if not os.path.exists(path):
                os.mkdir(path)

def save(path: Path, data: object, ignore_ext: bool = False) -> Path:
    """Saves data to this path. Extension and saving function is determined from the type.
    If the correct extension was already in the path its also ok.
    At the moment we handle:
    - pyplot figures -> .pdf
    - dictionaries -> .yaml
    - list -> .yaml
    - numpy -> .npy
    - pandas dataframes -> .tsv
    - anndata -> .h5ad
    - strings -> .txt
    - _anything else_ -> .p (pickled with `dill`)
    Parameters
    ----------
    path : Path
        The full path to save to
    data: object
        Data to save
    ignore_ext : bool
        Whether to ignore adding the normal expected extension
    Returns
    -------
    Path
        The final path to the file
    """
    if not isinstance(path, Path):
        path = Path(path)

    # Make sure the folder exists:
    path.parent.mkdir(parents=True, exist_ok=True)

    annotation_path = os.path.dirname(os.path.abspath(__file__))
    with open(annotation_path+"/tcga_anndata_groupings.yaml", 'r') as stream:
        tcga_annotations = yaml.safe_load(stream)

    def make_path(p: Path, ext: str) -> Path:
        """If the path doesn't end with the given extension add the extension to the path.
        Parameters
        ----------
        p : Path
            The path
        ext : str
            The expected extension
        Returns
        -------
        Path
            The fixed path
        """
        if not ignore_ext and not p.name.endswith(ext):
            return p.parent.joinpath(f"{p.name}{ext}")
        return p


    # PyPlot Figure
    if isinstance(data, mpl.figure.Figure):
        path = make_path(path, ".pdf")
        data.savefig(path)
        plt.close(data)
    # Dict ⇒ YAML Files
    elif isinstance(data, dict):
        path = make_path(path, ".yaml")
        with open(path, "w") as fp:
            yaml.dump(data, fp)
    # List ⇒ YAML Files
    elif isinstance(data, list):
        path = make_path(path, ".yaml")
        with open(path, "w") as fp:
            yaml.dump(data, fp)
    # NumPy Array
    elif isinstance(data, np.ndarray):
        path = make_path(path, ".npy")
        np.save(path, data)
    # Dataframes ⇒ TSV
    elif isinstance(data, pd.DataFrame):
        path = make_path(path, ".tsv")
        data.to_csv(path, sep="\t")
    # AnnData
    elif isinstance(data, anndata.AnnData):
        path = make_path(path, ".h5ad")
        for date_col in set(tcga_annotations['anndata']['obs']['datetime_columns']) & set(data.obs.columns):
            if "datetime" in data.obs[date_col].dtype.name:
                data.obs[date_col] = data.obs[date_col].dt.strftime("%Y-%m-%d")
            else:
                logger.info(f"Column {date_col} in obs should be a date but isnt formatted as one.")
        data.write(path)
    # Strings to normal files
    elif isinstance(data, str):
        path = make_path(path, ".txt")
        with open(path, "w") as fp:
            fp.write(data)
    # Everything else ⇒ pickle
    else:
        path = make_path(path, ".p")
        dill.dump(data, open(path, "wb"))
    return path



def _resolve_path(path: Path) -> Path:
    """Given a path, will try to resolve it in multiple ways:

    1. Is it a path to a S3 bucket?
    2. Is it a global/local file that exists?
    3. Is it path that is a prefix to a file that is unique?

    Parameters
    ----------
    path : Path
        The path

    Returns
    -------
    Path
        The global resolved file.

    Raises
    ------
    FileNotFoundError
        If the file doesn't exists or if there are multiple files that match the glob.
    """
    if not path.name.startswith("/"):
        path = path.expanduser().resolve()

    # If it exists we'll take it:
    if path.exists():
        return path

    # But mostly we load files without the extension so we glob for a uniue file:
    glob_name = path.name if path.name.endswith("*") else path.name + "*"
    paths = list(path.parent.glob(glob_name))
    if len(paths) == 1:
        return paths[0]  # was unique glob

    raise FileNotFoundError(
        f"Was trying to resolve path\n\t{path}*\nbut was ambigious because there are no or multiple files that fit the glob."
    )

def _to_int_string(element: Any) -> str:
    """Casts a number to a fixed formatted string that's nice categoriazebale.

    Parameters
    ----------
    element : Any
        The number, float or int

    Returns
    -------
    str
        Either the number formatted as a string or the original input if it
        didn't work
    """
    try:
        fl = float(element)
        return f"{fl:0.0f}"
    except:
        return element

def cast_anndata(ad: AnnData) -> None:
    """Fixes the data-type in the `.obs` and `.var` DataFrame columns of an
    AnnData object. __Works in-place__. Currently does the following:

    1.1. Enforces numerical-categorical `.obs` columns
    1.2. Makes all other `.obs` columns categoricals
    1.3. Makes date-time `.obs` columns, non-categorical pandas `datetime64`
    1.4. Enforces real strinng `.obs` columns, to be strings not categoricals
    1.5. Enforces some numerical `.obs` columns

    Configuration for which column belongs in which group is configured in
    `/transforna/utils/ngs_annotations.yaml` in this repository.

    Parameters
    ----------
    ad : AnnData
        The AnnData object
    """
    # 1. Fix obs-annotation dtypes

    # 1.1. Force numerical looking columns to be actual categorical variables
    annotation_path = os.path.dirname(os.path.abspath(__file__))
    with open(annotation_path+"/tcga_anndata_groupings.yaml", 'r') as stream:
        tcga_annotations = yaml.safe_load(stream)
    numerical_categorical_columns: List[str] = set(tcga_annotations['anndata']['obs']['numerical_categorical_columns']) & set(
        ad.obs.columns
    )
    for column in numerical_categorical_columns:
        ad.obs[column] = ad.obs[column].apply(_to_int_string).astype("U").astype("category")

    # 1.2. Forces string and mixed columns to be categoricals
    ad.strings_to_categoricals()

    # 1.3. DateTime, parse dates from string
    datetime_columns: List[str] = set(tcga_annotations['anndata']['obs']['datetime_columns']) & set(ad.obs.columns)
    for column in datetime_columns:
        try:
            ad.obs[column] = pd.to_datetime(ad.obs[column]).astype("datetime64[ns]")
        except ValueError as e:
            warning(
                f"""to_datetime error (parsing "unparseable"):\n {e}\nColumn
                {column} will be set as string not as datetime."""
            )
            ad.obs[column] = ad.obs[column].astype("string")

    # 1.4. Make _real_ string columns to force to be string, reversing step 1.2.
    # These are columns that contain acutal text, something like an description
    # or also IDs, which are identical not categories.
    string_columns: List[str] = set(tcga_annotations['anndata']['obs']['string_columns']) & set(ad.obs.columns)
    for column in string_columns:
        ad.obs[column] = ad.obs[column].astype("string")

    # 1.5. Force numerical columns to be numerical, this is necesary with some
    # invalid inputs or NaNs
    numerical_columns: List[str] = set(tcga_annotations['anndata']['obs']['numerical_columns']) & set(ad.obs.columns)
    for column in numerical_columns:
        ad.obs[column] = pd.to_numeric(ad.obs[column], errors="coerce")

    # 2. Fix var-annotation dtypes

    # 2.1. Enforce boolean columns to be real python bools, normally NaNs become
    # True here, which we change to False.
    boolean_columns: List[str] = set(tcga_annotations['anndata']['var']['boolean_columns']) & set(ad.var.columns)
    for column in boolean_columns:
        ad.var[column].fillna(False, inplace=True)
        ad.var[column] = ad.var[column].astype(bool)


def load(path: str, ext: str = None, **kwargs):
    """Loads the given filepath.

    This will use the extension of the filename to determine what to use for
    reading (if not overwritten). Most common use-case:

    At the moment we handle:

    - pickled objects (.p)
    - numpy objects (.npy)
    - dataframes (.csv, .tsv)
    - json files (.json)
    - yaml files (.yaml)
    - anndata files (.h5ad)
    - excel files (.xlsx)
    - text (.txt)

    Parameters
    ----------
    path : str
        The file-name of the cached file, without extension. (Or path)
        The file-name can be a glob match e.g. `/data/something/LC__*__21.7.2.*`
        which matches the everything with anything filling the stars. This only 
        works if there is only one match. So this is shortcut if you do not know
        the full name but you know there is only one.
    ext : str, optional
        The extension to assume, ignoring the actual extension. E.g. loading
        "tsv" for a "something.csv" file with tab-limits, by default None

    Returns
    -------
    Whatever is in the saved file.

    Raises
    ------
    FileNotFoundError
        If a given path doesn't exist or doesn't give a unqiue file path.
    NotImplementedError
        Trying to load a file with an extension we do not have loading code for.
    """
    path = _resolve_path(Path(path))

    # If extension is not overwritten take the one from the path_
    if ext is None:
        ext = path.suffix[1:]

    # Pickle files
    if ext == "p":
        return pickle.load(open(path, "rb"))
    # Numpy Arrays
    elif ext == "npy":
        return np.load(path)
    # TSV ⇒ DataFrame
    elif ext == "tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    # CSV ⇒ DataFrame
    elif ext == "csv":
        return pd.read_csv(path, **kwargs)
    # JSON ⇒ dict
    elif ext == "json":
        return json.load(open(path))
    # YAML ⇒ dict
    elif ext == "yaml":
        return yaml.load(open(path), Loader=yaml.SafeLoader)
    # AnnData
    elif ext == "h5ad":
        ad = anndata.read_h5ad(path, **kwargs)
        cast_anndata(ad)
        return ad
    # Excel files ⇒ DataFrame
    elif ext == "xlsx":
        return pd.read_excel(path, **kwargs)
    # General text files ⇒ string
    elif ext == "txt":
        with open(path, "r") as text_file:
            return text_file.read()
    #fasta
    elif ext == "fa":
        ## load sequences
        with open(path) as fasta_file:
            identifiers = []
            seqs = []
            for title, sequence in SimpleFastaParser(fasta_file):
                identifiers.append(title.split(None, 1)[0])
                seqs.append(sequence)
        #convert sequences to dataframe
        return pd.DataFrame({'Sequences':seqs})
    else:
        raise NotImplementedError
