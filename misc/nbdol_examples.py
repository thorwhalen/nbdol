"""Comprehensive examples for using nbdol.

This file demonstrates various patterns for creating and manipulating
Jupyter notebooks using the nbdol library.
"""

from nbdol.base import (
    Notebook,
    NotebookStore,
    CellTemplates,
    notebook_from_metadata,
    populate_notebook,
)
import nbformat


# ============================================================================
# EXAMPLE 1: Basic Notebook Creation
# ============================================================================


def example_basic_creation():
    """Create a simple notebook from scratch."""
    nb = Notebook()

    # Add cells using convenience methods
    nb.append_markdown("# My Analysis")
    nb.append_markdown("This is a data analysis notebook.")
    nb.append_code("import pandas as pd\nimport numpy as np")
    nb.append_code("df = pd.read_csv('data.csv')")
    nb.append_markdown("## Exploration")
    nb.append_code("df.head()")

    # Access cells like a list
    print(f"Number of cells: {len(nb)}")
    print(f"First cell type: {nb[0]['cell_type']}")

    # Save to file
    nb.save('basic_analysis.ipynb')

    return nb


# ============================================================================
# EXAMPLE 2: Using NotebookStore (Dict-like File Management)
# ============================================================================


def example_notebook_store():
    """Manage notebooks as a dict-like collection."""
    store = NotebookStore('my_notebooks/')

    # Create and save notebooks using dict syntax
    nb1 = Notebook()
    nb1.append_markdown("# Notebook 1")
    store['analysis_01'] = nb1

    nb2 = Notebook()
    nb2.append_markdown("# Notebook 2")
    store['analysis_02'] = nb2

    # List all notebooks
    print("Available notebooks:", list(store))

    # Check existence
    if 'analysis_01' in store:
        print("Found analysis_01")

    # Load and modify
    nb = store['analysis_01']
    nb.append_code("# Additional code")
    store['analysis_01'] = nb  # Save changes

    # Delete a notebook
    # del store['analysis_02']

    return store


# ============================================================================
# EXAMPLE 3: Creating from Metadata (cosmodata integration)
# ============================================================================


def example_from_metadata():
    """Generate notebook from metadata dict."""
    # Simulate cosmodata metadata
    metadata = {
        'title': 'Bitcoin Trading Data',
        'description': 'Historical Bitcoin prices and trading volumes',
        'src': 'https://example.com/bitcoin.parquet',
        'target_filename': 'bitcoin.parquet',
        'ext': None,
        'install': 'cosmograph tabled cosmodata pandas',
        'installs_not_to_import': ['cosmograph'],
        'imports': 'from functools import partial\nfrom cosmograph import cosmo',
        'peep_mode': 'short',
        'peep_exclude_cols': [],
    }

    # Create notebook with default templates
    nb = notebook_from_metadata(
        metadata, template_sequence=['intro', 'setup', 'load', 'explore']
    )

    # Add custom sections
    nb.append_markdown("## Visualization")
    nb.append_code("# Add visualization code here")

    nb.save('bitcoin_analysis.ipynb')

    return nb


# ============================================================================
# EXAMPLE 4: Custom Templates
# ============================================================================


def example_custom_templates():
    """Create and use custom cell templates."""
    templates = CellTemplates()

    # Define a custom template
    def my_intro_template(meta):
        """Custom introduction template."""
        yield nbformat.v4.new_markdown_cell(f"# {meta['title']}")
        yield nbformat.v4.new_markdown_cell(
            f"*Author: {meta.get('author', 'Unknown')}*"
        )
        yield nbformat.v4.new_markdown_cell("---")

        if 'description' in meta:
            yield nbformat.v4.new_markdown_cell(f"## Overview\n\n{meta['description']}")

    def my_imports_template(meta):
        """Custom imports template."""
        packages = meta.get('packages', ['pandas', 'numpy'])
        import_lines = '\n'.join(f"import {pkg}" for pkg in packages)
        yield nbformat.v4.new_code_cell(import_lines)

    # Register templates
    templates.register('custom_intro', my_intro_template)
    templates.register('custom_imports', my_imports_template)

    # Use custom templates
    metadata = {
        'title': 'My Analysis',
        'author': 'Data Scientist',
        'description': 'Analysis of interesting data',
        'packages': ['pandas', 'matplotlib', 'seaborn'],
    }

    nb = notebook_from_metadata(
        metadata,
        templates=templates,
        template_sequence=['custom_intro', 'custom_imports'],
    )

    nb.save('custom_template_notebook.ipynb')

    return nb


# ============================================================================
# EXAMPLE 5: Batch Generation from Multiple Datasets
# ============================================================================


def example_batch_generation():
    """Generate notebooks for multiple datasets."""
    # Simulate multiple datasets
    datasets = {
        'bitcoin': {
            'title': 'Bitcoin Data',
            'description': 'BTC price history',
            'src': 'https://example.com/bitcoin.parquet',
            'target_filename': 'bitcoin.parquet',
        },
        'ethereum': {
            'title': 'Ethereum Data',
            'description': 'ETH price history',
            'src': 'https://example.com/ethereum.parquet',
            'target_filename': 'ethereum.parquet',
        },
        'weather': {
            'title': 'Weather Data',
            'description': 'Historical weather records',
            'src': 'https://example.com/weather.csv',
            'target_filename': 'weather.csv',
        },
    }

    # Create store for output
    store = NotebookStore('generated_notebooks/')

    # Generate notebook for each dataset
    for key, metadata in datasets.items():
        nb = populate_notebook(
            metadata,
            template_sequence=['intro', 'setup', 'load', 'explore'],
            n_viz_cells=3,
        )
        store[key] = nb
        print(f"Generated notebook for {key}")

    return store


# ============================================================================
# EXAMPLE 6: Modifying Existing Notebooks
# ============================================================================


def example_modify_existing():
    """Load and modify existing notebooks."""
    store = NotebookStore('notebooks/')

    # Load existing notebook
    nb = store['analysis']

    # Insert cell at specific position
    nb.insert(2, nbformat.v4.new_markdown_cell("## New Section"))

    # Modify existing cell
    if nb[0]['cell_type'] == 'markdown':
        nb[0]['source'] = "# Updated Title"

    # Append new sections
    nb.append_markdown("## Additional Analysis")
    nb.append_code("# New analysis code")

    # Delete a cell
    # del nb[5]

    # Save changes
    store['analysis'] = nb

    return nb


# ============================================================================
# EXAMPLE 7: Working with Notebook as Sequence
# ============================================================================


def example_sequence_operations():
    """Demonstrate sequence protocol operations."""
    nb = Notebook()

    # Build notebook using list operations
    cells = [
        nbformat.v4.new_markdown_cell("# Title"),
        nbformat.v4.new_code_cell("import pandas as pd"),
        nbformat.v4.new_markdown_cell("## Analysis"),
        nbformat.v4.new_code_cell("df = pd.read_csv('data.csv')"),
    ]

    # Extend notebook with multiple cells
    nb.extend(cells)

    # Slice operations
    first_three = nb[:3]
    print(f"First 3 cells: {len(first_three)}")

    # Iterate over cells
    for i, cell in enumerate(nb):
        print(f"Cell {i}: {cell['cell_type']}")

    # Check cell types
    markdown_cells = [cell for cell in nb if cell['cell_type'] == 'markdown']
    code_cells = [cell for cell in nb if cell['cell_type'] == 'code']

    print(f"Markdown cells: {len(markdown_cells)}, Code cells: {len(code_cells)}")

    return nb


# ============================================================================
# EXAMPLE 8: Integration with cosmodata (realistic scenario)
# ============================================================================


def example_cosmodata_integration(metas, output_dir='cosmo_notebooks/notebooks/'):
    """Real-world example with cosmodata integration.

    Args:
        metas: The cosmodata.metas mapping object
        output_dir: Output directory for notebooks
    """
    store = NotebookStore(output_dir)

    # Generate notebooks for specific datasets
    datasets_to_generate = ['bitcoin', 'weather', 'covid']

    for dataset_key in datasets_to_generate:
        if dataset_key not in metas:
            print(f"Warning: {dataset_key} not found in metas")
            continue

        meta = metas[dataset_key]

        # Create notebook from metadata
        nb = populate_notebook(
            meta, template_sequence=['intro', 'setup', 'load', 'explore'], n_viz_cells=5
        )

        # Add dataset-specific customizations
        if dataset_key == 'bitcoin':
            nb.append_markdown("## Price Analysis")
            nb.append_code("# Analyze price trends\ndf['price'].plot()")
        elif dataset_key == 'weather':
            nb.append_markdown("## Temperature Analysis")
            nb.append_code("# Analyze temperature patterns\ndf['temperature'].hist()")

        # Save to store
        store[dataset_key] = nb
        print(f"Generated {dataset_key}.ipynb")

    return store


# ============================================================================
# EXAMPLE 9: Adding Common Sections to Multiple Notebooks
# ============================================================================


def example_batch_update():
    """Add a common section to multiple existing notebooks."""
    store = NotebookStore('notebooks/')

    # New section to add
    new_section = [
        nbformat.v4.new_markdown_cell("## Performance Metrics"),
        nbformat.v4.new_code_cell("import time\nstart = time.time()"),
        nbformat.v4.new_code_cell(
            "# ... analysis code ...\nprint(f'Elapsed: {time.time() - start:.2f}s')"
        ),
    ]

    # Add to all notebooks
    for key in list(store):
        nb = store[key]
        nb.extend(new_section)
        store[key] = nb
        print(f"Updated {key}")


# ============================================================================
# EXAMPLE 10: Export to Dict (for backwards compatibility)
# ============================================================================


def example_export_to_dict():
    """Export notebook to dict format."""
    nb = Notebook()
    nb.append_markdown("# Test")
    nb.append_code("x = 1")

    # Get as dict (compatible with old notebook_gen.py format)
    nb_dict = nb.to_dict()

    print(f"Notebook format: {nb_dict['nbformat']}")
    print(f"Number of cells: {len(nb_dict['cells'])}")

    # Can also access nbformat node directly
    nbformat_node = nb._nb

    return nb_dict


if __name__ == '__main__':
    print("Running nbdol examples...\n")

    # Run basic examples
    print("1. Basic creation")
    example_basic_creation()

    print("\n2. Notebook store")
    example_notebook_store()

    print("\n3. From metadata")
    example_from_metadata()

    print("\n4. Custom templates")
    example_custom_templates()

    print("\n5. Batch generation")
    example_batch_generation()

    print("\n7. Sequence operations")
    example_sequence_operations()

    print("\nAll examples completed!")
