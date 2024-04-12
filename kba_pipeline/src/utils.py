import pandas as pd
import os
import errno
from pathlib import Path
from Bio.SeqIO.FastaIO import SimpleFastaParser
from datetime import datetime
from getpass import getuser

import logging
from rich.logging import RichHandler
from functools import wraps
from time import perf_counter
from typing import Callable

default_path = '../outputs/'

def humanize_time(time_in_seconds: float, /) -> str:
    """Return a nicely human-readable string of a time_in_seconds.

    Parameters
    ----------
    time_in_seconds : float
        Time in seconds, (not full seconds).

    Returns
    -------
    str
        A description of the time in one of the forms:
        - 300.1 ms
        - 4.5 sec
        - 5 min 43.1 sec
    """
    sgn = "" if time_in_seconds >= 0 else "- "
    time_in_seconds = abs(time_in_seconds)
    if time_in_seconds < 1:
        return f"{sgn}{time_in_seconds*1e3:.1f} ms"
    elif time_in_seconds < 60:
        return f"{sgn}{time_in_seconds:.1f} sec"
    else:
        return f"{sgn}{int(time_in_seconds//60)} min {time_in_seconds%60:.1f} sec"


class log_time:
    """A decorator / context manager to log the time a certain function / code block took.

    Usage either with:

        @log_time(log)
        def function_getting_logged_every_time(…):
            …

    producing:

        function_getting_logged_every_time took 5 sec.

    or:

        with log_time(log, "Name of this codeblock"):
            …

    producing:

        Name of this codeblock took 5 sec.
    """

    def __init__(self, logger: logging.Logger, name: str = None):
        """
        Parameters
        ----------
        logger : logging.Logger
            The logger to use for logging the time, if None use print.
        name : str, optional
            The name in the message, when used as a decorator this defaults to the function name, by default None
        """
        self.logger = logger
        self.name = name

    def __call__(self, func: Callable):
        if self.name is None:
            self.name = func.__qualname__

        @wraps(func)
        def inner(*args, **kwds):
            with self:
                return func(*args, **kwds)

        return inner

    def __enter__(self):
        self.start_time = perf_counter()

    def __exit__(self, *exc):
        self.exit_time = perf_counter()

        time_delta = humanize_time(self.exit_time - self.start_time)
        if self.logger is None:
            print(f"{self.name} took {time_delta}.")
        else:
            self.logger.info(f"{self.name} took {time_delta}.")


def write_2_log(log_file):
    # Setup logging
    log_file_handler = logging.FileHandler(log_file)
    log_file_handler.setLevel(logging.INFO)
    log_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    log_rich_handler = RichHandler()
    log_rich_handler.setLevel(logging.INFO) #cli_args.log_level
    log_rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, datefmt="[%X]", handlers=[log_file_handler, log_rich_handler])


def fasta2df(path):
    with open(path) as fasta_file:
        identifiers = []
        seqs = []
        for header, sequence in SimpleFastaParser(fasta_file):
            identifiers.append(header)
            seqs.append(sequence)

    fasta_df = pd.DataFrame(seqs, identifiers, columns=['sequence'])
    fasta_df['sequence'] = fasta_df.sequence.apply(lambda x: x.replace('U','T'))
    return fasta_df



def fasta2df_subheader(path, id_pos):
    with open(path) as fasta_file:
        identifiers = []
        seqs = []
        for header, sequence in SimpleFastaParser(fasta_file):
            identifiers.append(header.split(None)[id_pos])
            seqs.append(sequence)

    fasta_df = pd.DataFrame(seqs, identifiers, columns=['sequence'])
    fasta_df['sequence'] = fasta_df.sequence.apply(lambda x: x.replace('U','T'))
    return fasta_df



def build_bowtie_index(bowtie_index_file):
    #index_example = Path(bowtie_index_file + '.1.ebwt')
    #if not index_example.is_file():
    print('-------- index is build --------')
    os.system(f"bowtie-build {bowtie_index_file + '.fa'} {bowtie_index_file}")
    #else: print('-------- previously built index is used --------')



def make_output_dir(fasta_file):
    output_dir = default_path + datetime.now().strftime('%Y-%m-%d') + ('__') + fasta_file.replace('.fasta', '').replace('.fa', '') + '/'
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise # This was not a "directory exist" error..
    return output_dir


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join([complement[base] for base in seq[::-1]])

