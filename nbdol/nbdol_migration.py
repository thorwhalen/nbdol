"""Example of migrating from old notebook_gen.py to nbdol.

This shows how to refactor the original NotebookParams-based approach
to use the new nbdol module with built-in protocols.
"""

from typing import Any, Mapping, Optional

from nbdol.base import Notebook, NotebookStore, populate_notebook

_DEFAULT_METADATA_VALUES: dict[str, Any] = {
    "ext": None,
    "install": "cosmograph tabled cosmodata",
    "installs_not_to_import": ["cosmograph"],
    "imports": """from functools import partial 
from cosmograph import cosmo""",
    "viz_columns_info": None,
    "related_code": None,
    "peep_mode": "short",
    "peep_exclude_cols": [],
}

_REQUIRED_BASE_FIELDS = ("src", "target_filename")
_REQUIRED_TEXT_FIELDS = ("title", "description")


def _metadata_with_defaults(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Merge metadata with defaults and support legacy field names."""
    merged: dict[str, Any] = {**_DEFAULT_METADATA_VALUES, **dict(metadata)}

    if "title" not in merged and "dataset_name" in merged:
        merged["title"] = merged["dataset_name"]
    if "description" not in merged and "dataset_description" in merged:
        merged["description"] = merged["dataset_description"]

    missing_base = [field for field in _REQUIRED_BASE_FIELDS if field not in merged]
    if missing_base:
        raise ValueError(
            f"Metadata must include {', '.join(_REQUIRED_BASE_FIELDS)}; "
            f"missing: {', '.join(missing_base)}"
        )

    missing_text = [field for field in _REQUIRED_TEXT_FIELDS if field not in merged]
    if missing_text:
        raise ValueError(
            "Metadata must include 'title' and 'description' (or the legacy "
            "'dataset_name'/'dataset_description' aliases)."
        )

    if isinstance(merged.get("installs_not_to_import"), list):
        merged["installs_not_to_import"] = list(merged["installs_not_to_import"])
    if isinstance(merged.get("peep_exclude_cols"), list):
        merged["peep_exclude_cols"] = list(merged["peep_exclude_cols"])

    return merged


def create_notebook(
    metadata: Mapping[str, Any],
    *,
    output_path: Optional[str] = None,
    n_viz_cells: int = 5,
) -> Notebook:
    """Generate a Jupyter notebook from metadata using nbdol.

    Args:
        metadata: Mapping describing the notebook. Provide at least 'src',
            'target_filename', and either 'title'/'description' or their legacy
            aliases 'dataset_name'/'dataset_description'.
        output_path: Optional path to save the notebook.
        n_viz_cells: Number of empty visualization cells to create.

    Returns:
        Notebook instance (can be further modified)

    Examples:
        >>> metadata = {
        ...     'title': 'Test Dataset',
        ...     'description': 'A test dataset',
        ...     'src': 'https://example.com/data.csv',
        ...     'target_filename': 'data.csv',
        ... }
        >>> nb = create_notebook(metadata)
        >>> isinstance(nb, Notebook)
        True
    """
    metadata_with_defaults = _metadata_with_defaults(metadata)

    nb = populate_notebook(
        metadata_with_defaults,
        template_sequence=("intro", "setup", "load", "explore"),
        n_viz_cells=n_viz_cells,
        output_path=output_path,
    )

    return nb


def generate_notebooks_from_cosmodata(
    metas, *, output_dir: str = "notebooks/", dataset_keys: Optional[list[str]] = None
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
            meta, template_sequence=("intro", "setup", "load", "explore"), n_viz_cells=5
        )

        # Save using store (dict-like interface)
        store[key] = nb

    return store


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
        meta, template_sequence=("intro", "setup", "load", "explore")
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


if __name__ == "__main__":
    metadata = {
        "title": "Bitcoin Price Data",
        "description": "Historical Bitcoin prices and trading volume",
        "src": "https://example.com/bitcoin.parquet",
        "target_filename": "bitcoin.parquet",
    }

    # Create notebook
    nb = create_notebook(metadata, output_path="bitcoin_analysis.ipynb")

    # Can still modify after creation
    nb.append_markdown("## Additional Analysis")
    nb.append_code("# Custom code here")
    nb.save()  # Saves to original path

    print(f"Created notebook with {len(nb)} cells")
