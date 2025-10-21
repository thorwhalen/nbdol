"""Example of migrating from old notebook_gen.py to nbdol.

This shows how to refactor the original NotebookParams-based approach
to use the new nbdol module with built-in protocols.
"""

from dataclasses import dataclass, field
from typing import Optional
from nbdol.base import Notebook, NotebookStore, populate_notebook


# Keep the same NotebookParams for backwards compatibility
@dataclass
class NotebookParams:
    """Parameters for generating a Jupyter notebook.

    Examples:
        >>> params = NotebookParams(
        ...     src='https://example.com/data.csv',
        ...     target_filename='data.csv',
        ...     dataset_name='Test Dataset',
        ...     dataset_description='A test dataset'
        ... )
    """

    src: str
    target_filename: str
    dataset_name: str
    dataset_description: str
    ext: Optional[str] = None
    install: str = "cosmograph tabled cosmodata"
    installs_not_to_import: list[str] = field(default_factory=lambda: ["cosmograph"])
    imports: str = field(
        default_factory=lambda: """from functools import partial 
from cosmograph import cosmo"""
    )
    viz_columns_info: Optional[str] = None
    related_code: Optional[str] = None
    peep_mode: str = "short"
    peep_exclude_cols: list[str] = field(default_factory=list)


# NEW APPROACH: Simple wrapper that converts params to dict
def create_notebook(
    params: NotebookParams, *, output_path: Optional[str] = None, n_viz_cells: int = 5
) -> Notebook:
    """Generate a Jupyter notebook from parameters using nbdol.

    This is a drop-in replacement for the old create_notebook function,
    but returns a Notebook object instead of a dict.

    Args:
        params: NotebookParams instance with all configuration
        output_path: Optional path to save the notebook
        n_viz_cells: Number of empty visualization cells to create

    Returns:
        Notebook instance (can be further modified)

    Examples:
        >>> params = NotebookParams(
        ...     src='https://example.com/data.csv',
        ...     target_filename='data.csv',
        ...     dataset_name='Test',
        ...     dataset_description='Test data'
        ... )
        >>> nb = create_notebook(params)
        >>> nb.append_markdown("## Custom Section")
        >>> nb.save('output.ipynb')
    """
    # Convert dataclass to dict for metadata
    metadata = {
        'title': params.dataset_name,
        'description': params.dataset_description,
        'src': params.src,
        'target_filename': params.target_filename,
        'ext': params.ext,
        'install': params.install,
        'installs_not_to_import': params.installs_not_to_import,
        'imports': params.imports,
        'viz_columns_info': params.viz_columns_info,
        'related_code': params.related_code,
        'peep_mode': params.peep_mode,
        'peep_exclude_cols': params.peep_exclude_cols,
    }

    # Use nbdol to create the notebook
    nb = populate_notebook(
        metadata,
        template_sequence=('intro', 'setup', 'load', 'explore'),
        n_viz_cells=n_viz_cells,
        output_path=output_path,
    )

    return nb


# For backwards compatibility: return dict
def create_notebook_dict(
    params: NotebookParams, *, output_path: Optional[str] = None, n_viz_cells: int = 5
) -> dict:
    """Generate notebook and return as dict (old API).

    Examples:
        >>> params = NotebookParams(  # doctest: +SKIP
        ...     src='https://example.com/data.csv',
        ...     target_filename='data.csv',
        ...     dataset_name='Test Dataset',
        ...     dataset_description='A test dataset'
        ... )
        >>> nb_dict = create_notebook_dict(params)  # doctest: +SKIP
        >>> nb_dict['nbformat']  # doctest: +SKIP
        4
    """
    nb = create_notebook(params, output_path=output_path, n_viz_cells=n_viz_cells)
    return nb.to_dict()


# EXAMPLE: Working with cosmodata directly
def generate_notebooks_from_cosmodata(
    metas, *, output_dir: str = 'notebooks/', dataset_keys: Optional[list[str]] = None
) -> NotebookStore:
    """Generate notebooks for multiple datasets from cosmodata.

    Args:
        metas: cosmodata.metas mapping object
        output_dir: Directory to save notebooks
        dataset_keys: Optional list of specific dataset keys to generate.
                     If None, generates for all datasets.

    Returns:
        NotebookStore containing all generated notebooks

    Examples:
        >>> from cosmodata import metas  # doctest: +SKIP
        >>> store = generate_notebooks_from_cosmodata(  # doctest: +SKIP
        ...     metas,
        ...     dataset_keys=['bitcoin', 'weather']
        ... )
        >>> 'bitcoin' in store  # doctest: +SKIP
        True
    """
    store = NotebookStore(output_dir)

    keys_to_process = dataset_keys if dataset_keys else list(metas.keys())

    for key in keys_to_process:
        meta = metas[key]

        # cosmodata meta dict can be used directly as metadata
        nb = populate_notebook(
            meta, template_sequence=('intro', 'setup', 'load', 'explore'), n_viz_cells=5
        )

        # Save using store (dict-like interface)
        store[key] = nb

    return store


# EXAMPLE: Custom template for specific use case
def create_custom_analysis_notebook(
    dataset_key: str,
    metas,
    custom_analysis_code: str,
    *,
    output_path: Optional[str] = None,
) -> Notebook:
    """Create notebook with custom analysis section.

    Examples:
        >>> nb = create_custom_analysis_notebook(  # doctest: +SKIP
        ...     'bitcoin',
        ...     metas,
        ...     custom_analysis_code='df.plot()'
        ... )
    """
    meta = metas[dataset_key]

    # Create base notebook
    nb = populate_notebook(
        meta, template_sequence=('intro', 'setup', 'load', 'explore')
    )

    # Add custom analysis section
    nb.append_markdown("## Custom Analysis")
    nb.append_code(custom_analysis_code)

    # Add visualization cells
    nb.append_markdown("## Visualizations")
    for _ in range(3):
        nb.append_code("")

    if output_path:
        nb.save(output_path)

    return nb


# EXAMPLE: Modifying existing notebooks
def add_section_to_existing_notebooks(
    store: NotebookStore,
    section_title: str,
    section_code: str,
    *,
    notebook_keys: Optional[list[str]] = None,
) -> None:
    """Add a new section to existing notebooks in a store.

    Args:
        store: NotebookStore instance
        section_title: Title for the new section
        section_code: Code to add in the new section
        notebook_keys: Optional list of specific notebooks to modify

    Examples:
        >>> store = NotebookStore('notebooks/')
        >>> add_section_to_existing_notebooks(
        ...     store,
        ...     "## Statistical Analysis",
        ...     "df.describe()"
        ... )
    """
    keys = notebook_keys if notebook_keys else list(store)

    for key in keys:
        # Load notebook (dict-like access)
        nb = store[key]

        # Add new section
        nb.append_markdown(section_title)
        nb.append_code(section_code)

        # Save back (dict-like assignment)
        store[key] = nb


if __name__ == '__main__':
    # Example usage
    params = NotebookParams(
        src='https://example.com/bitcoin.parquet',
        target_filename='bitcoin.parquet',
        dataset_name='Bitcoin Price Data',
        dataset_description='Historical Bitcoin prices and trading volume',
    )

    # Create notebook
    nb = create_notebook(params, output_path='bitcoin_analysis.ipynb')

    # Can still modify after creation
    nb.append_markdown("## Additional Analysis")
    nb.append_code("# Custom code here")
    nb.save()  # Saves to original path

    print(f"Created notebook with {len(nb)} cells")
