"""Notebook Data Object Layer - Pythonic interface for Jupyter notebook manipulation.

This module provides a composable, functional approach to creating and modifying
Jupyter notebooks using standard Python protocols (Mapping, Sequence, Iterable).

Examples:
    >>> from nbdol import Notebook
    >>> nb = Notebook()
    >>> nb.append_markdown("# Analysis")
    >>> nb.append_code("import pandas as pd")
    >>> len(nb)
    2
"""

from nbdol.base import (
    Notebook,
    NotebookStore,
    CellTemplates,
    notebook_from_metadata,
    populate_notebook,
    markdown_cell,
    code_cell,
)

__all__ = [
    'Notebook',
    'NotebookStore',
    'CellTemplates',
    'notebook_from_metadata',
    'populate_notebook',
    'markdown_cell',
    'code_cell',
]
