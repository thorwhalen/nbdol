# nbdol - Notebook Data Object Layer

Pythonic interface for Jupyter notebook manipulation using standard Python protocols (Mapping, Sequence, Iterable).

## Overview

`nbdol` provides a composable, functional approach to creating and modifying Jupyter notebooks. It uses built-in Python protocols to make notebook manipulation feel natural and intuitive.

### Key Features

- **List-like Notebook interface**: Notebooks behave as `MutableSequence` of cells
- **Dict-like file storage**: Manage notebook files using `MutableMapping` interface
- **Template system**: Composable cell generators using `Mapping` protocol
- **Integration ready**: Designed for use with cosmodata and cosmograph
- **Type-safe**: Proper type hints throughout
- **Built on nbformat**: Leverages the standard Jupyter notebook format library

## Installation

```bash
pip install nbdol
```

## Quick Start

### Create a Simple Notebook

```python
from nbdol import Notebook

# Create notebook
nb = Notebook()
nb.append_markdown("# My Analysis")
nb.append_code("import pandas as pd")
nb.append_code("df = pd.read_csv('data.csv')")

# Access like a list
print(f"Number of cells: {len(nb)}")
print(f"First cell: {nb[0]['source']}")

# Save
nb.save('analysis.ipynb')
```

### Manage Notebooks as Files

```python
from nbdol import NotebookStore

# Dict-like interface to notebook directory
store = NotebookStore('notebooks/')

# Save notebooks
store['analysis'] = nb  # Creates notebooks/analysis.ipynb

# Load notebooks
nb = store['analysis']

# List notebooks
print(list(store))  # ['analysis', 'exploration', ...]

# Check existence
if 'analysis' in store:
    print("Found it!")

# Delete
del store['old_notebook']
```

### Generate from Metadata

```python
from nbdol import populate_notebook

# Metadata dict (e.g., from cosmodata)
metadata = {
    'title': 'Bitcoin Analysis',
    'description': 'BTC price history',
    'src': 'https://example.com/bitcoin.parquet',
    'target_filename': 'bitcoin.parquet'
}

# Generate notebook with templates
nb = populate_notebook(
    metadata,
    template_sequence=['intro', 'setup', 'load', 'explore'],
    n_viz_cells=5,
    output_path='bitcoin.ipynb'
)
```

## Core Components

### Notebook (MutableSequence)

Acts like a list of cells with convenience methods:

```python
nb = Notebook()

# Add cells
nb.append_markdown("# Title")
nb.append_code("x = 42")

# List operations
nb.insert(1, markdown_cell("## Section"))
del nb[0]
cell = nb[2]
nb[3] = new_cell

# Iterate
for cell in nb:
    print(cell['cell_type'])

# Slice
first_five = nb[:5]

# Length
print(len(nb))
```

### NotebookStore (MutableMapping)

Manage notebooks in a directory using dict syntax:

```python
store = NotebookStore('notebooks/')

# Dict operations
store['name'] = notebook  # Save
nb = store['name']        # Load
del store['name']         # Delete
'name' in store           # Check existence
list(store)               # List all keys
len(store)                # Count notebooks

# Iteration
for key in store:
    nb = store[key]
    # process notebook
```

### CellTemplates (Mapping)

Registry of template functions:

```python
from nbdol import CellTemplates
import nbformat

templates = CellTemplates()

# Define template
def my_template(meta):
    yield nbformat.v4.new_markdown_cell(f"# {meta['title']}")
    yield nbformat.v4.new_code_cell("import pandas as pd")

# Register
templates.register('intro', my_template)

# Use in notebook
nb.extend_from_template(templates['intro'], {'title': 'Test'})
```

## Integration with cosmodata

```python
from cosmodata import metas
from nbdol import NotebookStore, populate_notebook

# Create store
store = NotebookStore('cosmo_notebooks/notebooks/')

# Generate notebooks for datasets
for dataset_key in ['bitcoin', 'weather', 'covid']:
    meta = metas[dataset_key]
    
    nb = populate_notebook(
        meta,
        template_sequence=['intro', 'setup', 'load', 'explore'],
        n_viz_cells=5
    )
    
    # Add custom sections
    nb.append_markdown("## Custom Analysis")
    nb.append_code("# Your code here")
    
    # Save
    store[dataset_key] = nb
```

## Advanced Usage

### Custom Templates

```python
from nbdol import CellTemplates, Notebook
import nbformat

templates = CellTemplates()

@templates.register('custom_intro')
def custom_intro_template(meta):
    """Custom introduction with author info."""
    yield nbformat.v4.new_markdown_cell(f"# {meta['title']}")
    yield nbformat.v4.new_markdown_cell(f"*By {meta['author']}*")
    yield nbformat.v4.new_markdown_cell(f"**Date:** {meta['date']}")
    
    if 'description' in meta:
        yield nbformat.v4.new_markdown_cell(f"## Overview\n\n{meta['description']}")

# Use template
nb = Notebook()
nb.extend_from_template(
    templates['custom_intro'],
    {'title': 'Analysis', 'author': 'Jane Doe', 'date': '2024-01-01'}
)
```

### Batch Operations

```python
# Add section to multiple notebooks
store = NotebookStore('notebooks/')

for key in store:
    nb = store[key]
    nb.append_markdown("## New Section")
    nb.append_code("# New code")
    store[key] = nb  # Save changes
```

### Modify Existing Notebooks

```python
# Load existing
nb = Notebook.from_file('existing.ipynb')

# Modify
nb[0]['source'] = "# Updated Title"
nb.insert(2, nbformat.v4.new_markdown_cell("## New Section"))
nb.append_code("# Additional code")

# Save
nb.save()  # Saves to original path
```

## Design Philosophy

`nbdol` follows these principles:

1. **Use built-in protocols**: Leverage `MutableSequence`, `MutableMapping`, etc.
2. **Composability**: Mix and match functions and classes
3. **Type safety**: Proper type hints throughout
4. **Minimal abstraction**: Thin wrappers over nbformat
5. **Discoverability**: Standard Python patterns feel natural

## API Reference

### Notebook

```python
class Notebook(MutableSequence):
    def __init__(self, cells=None, *, path=None)
    def append_markdown(self, content: str) -> None
    def append_code(self, code: str) -> None
    def extend_from_template(self, template_func, metadata: dict) -> None
    @classmethod
    def from_file(cls, path) -> 'Notebook'
    def save(self, path=None) -> None
    def to_dict(self) -> dict
    
    @property
    def cells -> list
    @property
    def metadata -> dict
```

### NotebookStore

```python
class NotebookStore(MutableMapping):
    def __init__(self, root_path='.', *, extension='.ipynb')
    # Implements: __getitem__, __setitem__, __delitem__, 
    #             __iter__, __len__, __contains__
```

### CellTemplates

```python
class CellTemplates(Mapping):
    def __init__(self)
    def register(self, name: str, template_func: Callable) -> None
    # Implements: __getitem__, __iter__, __len__
```

### Utility Functions

```python
def notebook_from_metadata(
    metadata: dict,
    *,
    templates=None,
    template_sequence=None
) -> Notebook

def populate_notebook(
    metadata: Mapping,
    *,
    template_sequence=('intro', 'setup', 'load', 'explore'),
    templates=None,
    output_path=None,
    n_viz_cells=0
) -> Notebook

def markdown_cell(content: str) -> dict
def code_cell(code: str) -> dict
```

## Default Templates

The library includes default templates for data analysis notebooks:

- **`intro`**: Dataset title, description, source info
- **`setup`**: Data parameters, package installation, imports
- **`load`**: Data loading code (cosmodata integration)
- **`explore`**: Data exploration/inspection code

These work with metadata dicts from cosmodata.

## Requirements

- Python 3.10+
- nbformat >= 5.0.0

## License

MIT

## Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8
- All functions have docstrings with examples
- Tests pass
- Type hints are used