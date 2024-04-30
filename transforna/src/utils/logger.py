"""This sub-module contains methods for logging to the terminal in a
standardized and simple way."""
import inspect
import logging
import time
import typing as ty
import warnings
from functools import wraps
from typing import Any, Callable, List

import pandas as pd
from anndata.logging import anndata_logger
from rich import get_console, pretty, traceback
from rich.logging import RichHandler

logger, console = None, None


def _install() -> None:
    """This herlper singleton function installs some option around printing and
    logging globally. Mostly usese `rich` for logging and printing and setting
    some options for pandas outputs. This function is run on start-up, so no
    need to ever run again."""
    global logger, console

    console = get_console()
    pretty.install()  # Installs automatic pretty-printing in the REPL (`breakpoint()`)
    traceback.install()  # Installs pretty-print for error tracebacks
    # Will set-up a global transforna logger with an `rich` handler (pretty-print)
    handler = RichHandler()
    handler.KEYWORDS.extend(["LOAD", "SAVE", "RUN", "SERGIO", "CACHE"])
    logger = logging.getLogger("transforna")
    logger.addHandler(handler)

    # Reduce output from anndata
    anndata_logger.setLevel("WARNING")
    warnings.simplefilter(action="ignore", category=FutureWarning)  # isort:skip

    # Set global options for Pandas output
    pd.set_option("display.max_rows", 24)
    pd.set_option("float_format", "{:.2e}".format)
    pd.set_option("display.width", 120)


_install()

__all__ = [
    "logger",
    "print",
    "info",
    "warning",
    "error",
    "rule",
    "input",
    "log",
    "question",
]


def print(*args, **kwargs) -> None:
    """Just using `rich` ro print to console."""
    console.print(*args, **kwargs)


def info(*args, **kwargs) -> None:
    """Log to the standard transforna logger using `rich` on level `info`."""
    logger.info(*args, **kwargs)


def warning(*args, **kwargs) -> None:
    """Log to the standard transforna logger using `rich` on level `warning`."""
    logger.warning(*args, **kwargs)


def error(*args, **kwargs) -> None:
    """Log to the standard transforna logger using `rich` on level `error`."""
    logger.error(*args, **kwargs)


def rule(*args) -> None:
    """Print a horizontal ruler to the console using `rich`. The given text will
    be horizontally centered."""
    console.rule(*args)


def input(*args) -> str:
    """Use `rich` input method to get input from the user in the console."""
    return console.input(*args)


class log:
    """This class acts as a general execution logger. Can be used two ways:


    """
    def __init__(self, message: str = "", level: str = "RUN"):
        """
        Parameters
        ----------
        message : str, optional
            The message would be printed in the logging, by default ""
        level : str, optional
            The logging level to use, by default "RUN"
            can be RUN, LOAD, SAVE, INFO, WARNING, ERROR
        """
        self.level = level
        self.message = message

    def __call__(self, func: Callable):
        """When used as a function decorator it wil wrap the fucntion call in a
        timed context. If the first argument to the function and the first
        result element of the function both have a `.shape` attribute, the log
        message will also display the change in shape of that element.


        Parameters
        ----------
        func : Callable
            The wrapped function

        Returns
        -------
        Callable
            The decorator
        """
        name = func.__qualname__

        @wraps(func)
        def inner(*args, **kwargs) -> Any:
            """This is the actual function wrapper

            Returns
            -------
            Any
                The result fron the wrapped function
            """
            intro_shape, outro_shape = None, None
            if len(args) > 0:
                first_arg = args[1] if is_method(func) else args[0]
                if hasattr(first_arg, "shape"):
                    intro_shape = first_arg.shape

            with log(name) as status:
                results = func(*args, **kwargs)

                first_res = results[0] if isinstance(results, tuple) else results
                if hasattr(first_res, "shape"):
                    outro_shape = first_res.shape
                if intro_shape is not None and outro_shape is not None:
                    status.message += f" {intro_shape} → {outro_shape}"

            return results

        return inner

    def __enter__(self):
        """When entering the context, we start a timer to time the execution
        time."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """When exiting the context we calculate the duration since entering the
        context and print and info log statement containing the message and the
        duration time."""
        duration = time.perf_counter() - self.start_time
        duration = f"[{int(duration)}s] "
        info(f"{self.level.ljust(6)}{duration}{self.message}")
        return False


def question(message: str, options: List[str]) -> str:
    """Ask a question to the user.

    Parameters
    ----------
    message : str
        The question
    options : List[str]
        The legal answers

    Returns
    -------
    str
        The picked answer
    """
    answer = ""
    options_string = ", ".join(sorted(options))
    while answer not in options:
        answer = console.input(f"{message} {{[yellow]{options_string}[/yellow]}} : ")
    return answer


def is_method(func: ty.Callable) -> bool:
    """Returns whether a given callable is a bound method of a class. This is
    done by checking whether the first argument is named "self". So it is not
    a perfect check!!

    Parameters
    ----------
    func : ty.Callable
        The function

    Returns
    -------
    bool
        Whether func is a bound method
    """
    try:
        first_agrument_name = next(iter(inspect.signature(func).parameters.keys()))
    # In case of no arguments:
    except StopIteration:
        return False
    return first_agrument_name == "self"
