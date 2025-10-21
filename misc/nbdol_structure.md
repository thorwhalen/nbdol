# nbdol Package Structure

## Directory Layout

```
nbdol/
├── nbdol/
│   ├── __init__.py          # Public API exports
│   ├── base.py              # Core classes (Notebook, NotebookStore, CellTemplates)
│   ├── util.py              # Helper functions and default templates
│   └── tests/
│       ├── __init__.py
│       └── test_base.py     # Unit tests
├── misc/
│   ├── CHANGELOG.md         # Version history and major changes
│   └── examples.py          # Usage examples (optional)
├── README.md                # Documentation
├── setup.py                 # Package setup
├── migration_example.py     # Refactoring guide from old code
└── examples.py              # Comprehensive usage examples
```

## Module Contents

### `nbdol/__init__.py`
Main entry point with public API:
- `Notebook` - MutableSequence of cells
- `NotebookStore` - MutableMapping for file storage
- `CellTemplates` - Mapping for template registry
- `notebook_from_metadata()` - Create from metadata
- `populate_notebook()` - High-level creation function

### `nbdol/base.py`
Core classes implementing built-in protocols:
- `Notebook(MutableSequence)` - 200 lines
- `NotebookStore(MutableMapping)` - 100 lines  
- `CellTemplates(Mapping)` - 50 lines

### `nbdol/util.py`
Utility functions and default templates:
- `notebook_from_metadata()` - Create from dict
- `populate_notebook()` - High-level wrapper
- `markdown_cell()`, `code_cell()` - Cell factories
- `_get_default_templates()` - Built-in templates
- Default template functions (intro, setup, load, explore)

### `nbdol/tests/test_base.py`
Comprehensive test suite:
- `TestNotebook` - 10+ test methods
- `TestNotebookStore` - 8 test methods
- `TestCellTemplates` - 5 test methods
- `TestUtilFunctions` - 2 test methods

## Quick Reference

### Creating a Notebook

```python
from nbdol import Notebook

nb = Notebook()
nb.append_markdown("# Title")
nb.append_code("import pandas as pd")
nb.save('output.ipynb')
```

### Using NotebookStore

```python
from nbdol import NotebookStore

store = NotebookStore('notebooks/')
store['analysis'] = Notebook()  # Save
nb = store['analysis']          # Load
del store['old']                # Delete
list(store)                     # List all
```

### Template-Based Generation

```python
from nbdol import populate_notebook

metadata = {
    'title': 'Bitcoin Analysis',
    'src': 'https://example.com/bitcoin.parquet',
    'target_filename': 'bitcoin.parquet'
}

nb = populate_notebook(
    metadata,
    template_sequence=['intro', 'setup', 'load', 'explore'],
    n_viz_cells=5,
    output_path='bitcoin.ipynb'
)
```

### Custom Templates

```python
from nbdol import CellTemplates, Notebook
import nbformat

templates = CellTemplates()

def my_template(meta):
    yield nbformat.v4.new_markdown_cell(f"# {meta['title']}")
    yield nbformat.v4.new_code_cell("import pandas as pd")

templates.register('custom', my_template)

nb = Notebook()
nb.extend_from_template(templates['custom'], {'title': 'Test'})
```

## Integration with cosmodata

```python
from cosmodata import metas
from nbdol import NotebookStore, populate_notebook

store = NotebookStore('cosmo_notebooks/notebooks/')

for key in ['bitcoin', 'weather']:
    nb = populate_notebook(metas[key], n_viz_cells=5)
    store[key] = nb
```

## Key Design Decisions

1. **Built-in Protocols**: Used `MutableSequence`, `MutableMapping`, `Mapping` for familiarity
2. **nbformat Integration**: Leveraged existing library for notebook structure
3. **Composability**: Mixed functions and classes for flexibility
4. **Type Safety**: Comprehensive type hints throughout
5. **Minimal Abstraction**: Thin wrappers over nbformat.NotebookNode

## Dependencies

- **Required**: `nbformat >= 5.0.0`
- **Development**: `pytest`, `pytest-cov`
- **Python**: 3.10+

## Installation

```bash
# From source
cd nbdol/
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# Run tests
pytest nbdol/tests/
```

## Migration from notebook_gen.py

The old `notebook_gen.py` used direct dict manipulation. Key changes:

**Old approach:**
```python
def create_notebook(params: NotebookParams) -> dict:
    cells = [_create_cell(...), ...]
    return {"cells": cells, "metadata": {...}, ...}
```

**New approach:**
```python
def create_notebook(params: NotebookParams) -> Notebook:
    metadata = dataclass_to_dict(params)
    return populate_notebook(metadata, n_viz_cells=5)
```

Benefits:
- Returns `Notebook` object (can be modified further)
- Cleaner template system
- Dict-like file management via `NotebookStore`
- Type-safe operations
- Extensible template registry

See `migration_example.py` for complete refactoring guide.

## Future Enhancements

Potential additions:
- Additional default templates (ML workflows, reports)
- Cell search/filter capabilities
- Notebook diff/merge utilities
- Export to other formats (using nbconvert)
- Parameterization support (using papermill)
- Metadata extraction/indexing

## Philosophy

Following Python best practices:
- "Explicit is better than implicit"
- "Simple is better than complex"  
- "Readability counts"
- "There should be one obvious way to do it"

Using built-in protocols makes the API discoverable and familiar to Python developers.
