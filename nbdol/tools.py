"""Utility helpers built on top of nbdol primitives."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import nbformat
from nbformat import NotebookNode

from .base import Notebook

Cell = dict[str, Any]
CellFilter = Callable[[Cell, int], bool]
CellProcessor = Callable[[Cell, int], Optional[Cell]]


def process_notebook(
    notebook_path: Union[str, Path],
    *,
    cell_filter: CellFilter,
    cell_processor: CellProcessor,
    output: Union[bool, str, Path] = False,
) -> Notebook:
    """Apply a cell processor to each cell that matches ``cell_filter``.

    Args:
        notebook_path: Path to the notebook file to process.
        cell_filter: Callable invoked with ``(cell, index)`` to decide if the
            cell should be processed.
        cell_processor: Callable invoked with ``(cell, index)`` to transform the
            cell. It may mutate the input cell in place (return ``None``) or
            return a replacement cell as a dict/NotebookNode.
        output: Controls how the processed notebook is emitted.
            * ``False`` (default) returns the in-memory :class:`Notebook`.
            * ``True`` saves back to ``notebook_path``.
            * ``str``/``Path`` saves to the provided location.

    Returns:
        The processed :class:`Notebook` instance. It is always returned so callers
        can continue chaining transformations regardless of ``output``.
    """
    nb = Notebook.from_file(notebook_path)

    for index, cell in enumerate(nb):
        if not cell_filter(cell, index):
            continue

        processed = cell_processor(cell, index)
        if processed is not None:
            if isinstance(processed, NotebookNode):
                nb[index] = processed
            elif isinstance(processed, dict):
                nb[index] = nbformat.from_dict(processed)
            else:
                raise TypeError(
                    "cell_processor must return a dict, NotebookNode, or None when mutating in place"
                )

    if output is True:
        nb.save(notebook_path)
    elif isinstance(output, (str, Path)):
        nb.save(output)
    elif output not in (False, None):
        raise ValueError("output must be False, True, or a filesystem path")

    return nb


def _has_error_output(cell: Cell, _: int) -> bool:
    outputs = cell.get("outputs", [])
    return any(output.get("output_type") == "error" for output in outputs)


def _clear_error_outputs(cell: Cell, _: int) -> Cell:
    outputs = cell.get("outputs", [])
    cleaned_outputs = [
        output for output in outputs if output.get("output_type") != "error"
    ]

    if cleaned_outputs != outputs:
        cell["outputs"] = cleaned_outputs

    return None


remove_tracebacks = partial(
    process_notebook,
    cell_filter=_has_error_output,
    cell_processor=_clear_error_outputs,
)
remove_tracebacks.__doc__ = (
    "Remove error outputs (tracebacks) from a notebook.\n\n"
    "Args:\n"
    "    notebook_path: Path to the notebook file to process.\n"
    "    output: See `process_notebook` for behaviour.\n\n"
    "Returns:\n"
    "    The processed Notebook instance."
)
