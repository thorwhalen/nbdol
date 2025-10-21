"""Base classes for notebook manipulation using built-in protocols.

This module provides the core abstractions for working with Jupyter notebooks
as native Python data structures.
"""

from collections.abc import MutableSequence, MutableMapping, Mapping, Iterator
from pathlib import Path
from typing import Any, Callable, Optional, Union
import nbformat
from nbformat.notebooknode import NotebookNode


class Notebook(MutableSequence):
    """A Jupyter notebook represented as a mutable sequence of cells.

    Wraps nbformat.NotebookNode and provides a list-like interface to cells.

    Examples:
        >>> nb = Notebook()
        >>> nb.append_markdown("# Title")
        >>> nb.append_code("x = 42")
        >>> len(nb)
        2
    """

    def __init__(
        self,
        cells: Optional[list[dict]] = None,
        *,
        path: Optional[Union[str, Path]] = None,
    ):
        """Initialize notebook with optional cells and path.

        Args:
            cells: Optional list of cell dicts to initialize with
            path: Optional path for saving (set when loaded from file)
        """
        if cells:
            self._nb = nbformat.v4.new_notebook(cells=cells)
        else:
            self._nb = nbformat.v4.new_notebook()
        self._path = Path(path) if path else None

    def __len__(self) -> int:
        """Return number of cells."""
        return len(self._nb.cells)

    def __getitem__(self, index: Union[int, slice]) -> Union[dict, list[dict]]:
        """Get cell(s) by index.

        Examples:
            >>> nb = Notebook()
            >>> nb.append_markdown("# Test")
            >>> nb[0]['cell_type']
            'markdown'
        """
        return self._nb.cells[index]

    def __setitem__(
        self, index: Union[int, slice], value: Union[dict, list[dict]]
    ) -> None:
        """Set cell(s) by index."""
        self._nb.cells[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Delete cell(s) by index."""
        del self._nb.cells[index]

    def insert(self, index: int, value: dict) -> None:
        """Insert cell at index."""
        self._nb.cells.insert(index, value)

    def append_markdown(self, content: str) -> None:
        """Append a markdown cell.

        Examples:
            >>> nb = Notebook()
            >>> nb.append_markdown("# Title")
            >>> nb[-1]['cell_type']
            'markdown'
        """
        cell = nbformat.v4.new_markdown_cell(content)
        self._nb.cells.append(cell)

    def append_code(self, code: str) -> None:
        """Append a code cell.

        Examples:
            >>> nb = Notebook()
            >>> nb.append_code("x = 42")
            >>> nb[-1]['cell_type']
            'code'
        """
        cell = nbformat.v4.new_code_cell(code)
        self._nb.cells.append(cell)

    def extend_from_template(
        self, template_func: Callable[[dict], Iterator[dict]], metadata: dict
    ) -> None:
        """Add cells from a template function using metadata.

        Args:
            template_func: Function that takes metadata and yields cell dicts
            metadata: Metadata to pass to template function

        Examples:
            >>> def my_template(meta):
            ...     yield nbformat.v4.new_markdown_cell(f"# {meta['title']}")
            >>> nb = Notebook()
            >>> nb.extend_from_template(my_template, {'title': 'Test'})
            >>> len(nb)
            1
        """
        for cell in template_func(metadata):
            self._nb.cells.append(cell)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'Notebook':
        """Load notebook from file.

        Examples:
            >>> nb = Notebook.from_file('analysis.ipynb')  # doctest: +SKIP
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            nb_node = nbformat.read(f, as_version=4)

        instance = cls.__new__(cls)
        instance._nb = nb_node
        instance._path = path
        return instance

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save notebook to file.

        Args:
            path: Path to save to. If None, uses self._path from loading/previous save.

        Examples:
            >>> nb = Notebook()
            >>> nb.append_markdown("# Test")
            >>> nb.save('output.ipynb')
        """
        save_path = Path(path) if path else self._path
        if save_path is None:
            raise ValueError("No path specified and notebook was not loaded from file")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            nbformat.write(self._nb, f)

        self._path = save_path

    def to_dict(self) -> dict:
        """Export as notebook JSON structure.

        Examples:
            >>> nb = Notebook()
            >>> d = nb.to_dict()
            >>> d['nbformat']
            4
        """
        # Convert NotebookNode to dict
        import json

        return json.loads(nbformat.writes(self._nb))

    @property
    def cells(self) -> list:
        """Direct access to cells list (for advanced usage)."""
        return self._nb.cells

    @property
    def metadata(self) -> dict:
        """Access notebook metadata."""
        return self._nb.metadata


class NotebookStore(MutableMapping):
    """File-based notebook storage with dict-like interface.

    Provides a mapping abstraction over a directory of notebook files.

    Examples:
        >>> store = NotebookStore('notebooks/')
        >>> nb = Notebook()
        >>> store['analysis'] = nb
        >>> 'analysis' in store
        True
    """

    def __init__(self, root_path: Union[str, Path] = '.', *, extension: str = '.ipynb'):
        """Initialize store at root_path.

        Args:
            root_path: Directory to store notebooks
            extension: File extension for notebooks
        """
        self._root = Path(root_path)
        self._ext = extension
        self._root.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        if not key.endswith(self._ext):
            key = f"{key}{self._ext}"
        return self._root / key

    def _path_to_key(self, path: Path) -> str:
        """Convert path to key (removes extension)."""
        return path.stem

    def __getitem__(self, key: str) -> Notebook:
        """Load notebook by key.

        Examples:
            >>> store = NotebookStore('notebooks/')
            >>> nb = store['analysis']
        """
        path = self._key_to_path(key)
        if not path.exists():
            raise KeyError(f"Notebook '{key}' not found at {path}")
        return Notebook.from_file(path)

    def __setitem__(self, key: str, notebook: Notebook) -> None:
        """Save notebook with key.

        Examples:
            >>> store = NotebookStore('notebooks/')
            >>> store['new'] = Notebook()
        """
        path = self._key_to_path(key)
        notebook.save(path)

    def __delitem__(self, key: str) -> None:
        """Delete notebook by key.

        Examples:
            >>> store = NotebookStore('notebooks/')  # doctest: +SKIP
            >>> del store['old']  # doctest: +SKIP
        """
        path = self._key_to_path(key)
        if not path.exists():
            raise KeyError(f"Notebook '{key}' not found at {path}")
        path.unlink()

    def __iter__(self) -> Iterator[str]:
        """Iterate over notebook keys.

        Examples:
            >>> store = NotebookStore('notebooks/')  # doctest: +SKIP
            >>> list(store)  # doctest: +SKIP
            ['analysis', 'exploration']
        """
        for path in self._root.glob(f"*{self._ext}"):
            yield self._path_to_key(path)

    def __len__(self) -> int:
        """Return number of notebooks in store."""
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        """Check if notebook exists."""
        if not isinstance(key, str):
            return False
        return self._key_to_path(key).exists()


class CellTemplates(Mapping):
    """Registry of cell template generators.

    Templates are functions that take metadata and yield cell dicts.

    Examples:
        >>> templates = CellTemplates()
        >>> def my_template(meta):
        ...     yield nbformat.v4.new_markdown_cell(f"# {meta['title']}")
        >>> templates.register('intro', my_template)
        >>> cells = list(templates['intro']({'title': 'Test'}))
    """

    def __init__(self):
        """Initialize empty template registry."""
        self._templates: dict[str, Callable] = {}

    def register(self, name: str, template_func: Callable) -> None:
        """Register a template function.

        Args:
            name: Name for the template
            template_func: Function that takes metadata and yields cells

        Examples:
            >>> templates = CellTemplates()
            >>> templates.register('intro', lambda m: [])
        """
        self._templates[name] = template_func

    def __getitem__(self, key: str) -> Callable:
        """Get template function by name."""
        return self._templates[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over template names."""
        return iter(self._templates)

    def __len__(self) -> int:
        """Return number of registered templates."""
        return len(self._templates)


from collections.abc import Mapping, Iterable
from typing import Optional, Union, Callable
from pathlib import Path
import nbformat

from nbdol.base import Notebook, CellTemplates


def notebook_from_metadata(
    metadata: dict,
    *,
    templates: Optional[CellTemplates] = None,
    template_sequence: Optional[Iterable[str]] = None,
) -> Notebook:
    """Create notebook from metadata dict using template sequence.

    Args:
        metadata: Metadata dict with notebook parameters
        templates: CellTemplates instance to use. If None, uses default templates.
        template_sequence: Sequence of template names to apply

    Returns:
        Populated Notebook instance

    Examples:
        >>> nb = notebook_from_metadata(
        ...     {'title': 'Bitcoin', 'src': 'http://...'},
        ...     template_sequence=['intro', 'setup']
        ... )
    """
    nb = Notebook()

    if templates is None:
        templates = _get_default_templates()

    if template_sequence is None:
        template_sequence = ['intro', 'setup', 'load', 'explore']

    for template_name in template_sequence:
        if template_name in templates:
            nb.extend_from_template(templates[template_name], metadata)

    return nb


def populate_notebook(
    metadata: Mapping,
    *,
    template_sequence: Iterable[str] = ('intro', 'setup', 'load', 'explore'),
    templates: Optional[CellTemplates] = None,
    output_path: Optional[Union[str, Path]] = None,
    n_viz_cells: int = 0,
) -> Notebook:
    """Higher-level function to create and optionally save populated notebook.

    Args:
        metadata: Metadata mapping with notebook parameters
        template_sequence: Sequence of template names to apply
        templates: CellTemplates instance to use
        output_path: Optional path to save notebook
        n_viz_cells: Number of empty visualization cells to add at end

    Returns:
        Populated Notebook instance

    Examples:
        >>> nb = populate_notebook(
        ...     {'title': 'Bitcoin'},
        ...     output_path='bitcoin.ipynb'
        ... )
    """
    nb = notebook_from_metadata(
        dict(metadata), templates=templates, template_sequence=template_sequence
    )

    if n_viz_cells > 0:
        nb.append_markdown("## Visualizations")
        for _ in range(n_viz_cells):
            nb.append_code("")

    if output_path:
        nb.save(output_path)

    return nb


def markdown_cell(content: str) -> dict:
    """Create a markdown cell dict.

    Examples:
        >>> cell = markdown_cell("# Title")
        >>> cell['cell_type']
        'markdown'
    """
    return nbformat.v4.new_markdown_cell(content)


def code_cell(code: str) -> dict:
    """Create a code cell dict.

    Examples:
        >>> cell = code_cell("x = 42")
        >>> cell['cell_type']
        'code'
    """
    return nbformat.v4.new_code_cell(code)


def _get_default_templates() -> CellTemplates:
    """Get default template registry for common notebook patterns.

    This creates templates suitable for data analysis notebooks using
    cosmodata and cosmograph.
    """
    templates = CellTemplates()

    def _intro_template(meta: dict) -> Iterable[dict]:
        """Generate introduction cells from metadata."""
        title = meta.get('title', meta.get('dataset_name', 'Dataset Analysis'))
        yield markdown_cell(f"# {title}")

        if desc := meta.get('description', meta.get('dataset_description')):
            yield markdown_cell(f"**Description:** {desc}")

        if src := meta.get('src'):
            filename = meta.get('target_filename', src.split('/')[-1].split('?')[0])
            yield markdown_cell(f"**Data Source:** [{filename}]({src})")

        if viz_info := meta.get('viz_columns_info'):
            yield markdown_cell(f"**Visualization notes:** {viz_info}")

        if related := meta.get('related_code'):
            yield markdown_cell(f"**Related code:** {related}")

    def _setup_template(meta: dict) -> Iterable[dict]:
        """Generate setup cells (parameters, installs, imports)."""
        yield markdown_cell("## Setup")
        yield markdown_cell("### Data Parameters")

        # Data parameters cell
        params_lines = []
        if ext := meta.get('ext'):
            params_lines.append(f"ext = {repr(ext)}")
        if src := meta.get('src'):
            params_lines.append(f"src = {repr(src)}")
        if target := meta.get('target_filename'):
            params_lines.append(f"target_filename = {repr(target)}")

        if params_lines:
            yield code_cell("\n".join(params_lines))

        # Install and import cell
        yield markdown_cell("### Install and Import")

        install_pkgs = meta.get('install', 'cosmograph tabled cosmodata')
        installs_not_to_import = meta.get('installs_not_to_import', ['cosmograph'])
        custom_imports = meta.get(
            'imports', 'from functools import partial\nfrom cosmograph import cosmo'
        )

        install_lines = [
            "import os",
            "if not os.getenv('IN_COSMO_DEV_ENV'):",
            f"    %pip install -q {install_pkgs}",
            "",
        ]

        # Import installed packages
        for pkg in install_pkgs.split():
            if pkg not in installs_not_to_import:
                install_lines.append(f"import {pkg}")

        if custom_imports:
            install_lines.append("")
            install_lines.append(custom_imports)

        yield code_cell("\n".join(install_lines))

    def _load_template(meta: dict) -> Iterable[dict]:
        """Generate data loading cells."""
        yield markdown_cell("## Load Data")

        load_code = """if ext:
    getter = partial(tabled.get_table, ext=ext)
else:
    getter = tabled.get_table

# acquire_data handles caching locally for faster future access
data = cosmodata.acquire_data(src, target_filename, getter=getter)"""

        yield code_cell(load_code)

    def _explore_template(meta: dict) -> Iterable[dict]:
        """Generate data exploration cells."""
        yield markdown_cell("## Explore Data")

        peep_mode = meta.get('peep_mode', 'short')
        peep_exclude = meta.get('peep_exclude_cols', [])

        peep_code = f"""mode = {repr(peep_mode)}  # Options: 'short', 'sample', 'stats'
exclude_cols = {repr(peep_exclude)}

cosmodata.print_dataframe_info(data, exclude_cols, mode=mode)"""

        yield code_cell(peep_code)

    templates.register('intro', _intro_template)
    templates.register('setup', _setup_template)
    templates.register('load', _load_template)
    templates.register('explore', _explore_template)

    return templates


# Convenience aliases
create_notebook = populate_notebook
