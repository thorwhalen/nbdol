# Changelog

## 2025-01-19 - Initial Release (v0.1.0)

### Core Implementation
- Implemented `Notebook` class as `MutableSequence` for list-like cell manipulation
- Implemented `NotebookStore` class as `MutableMapping` for dict-like file management
- Implemented `CellTemplates` class as `Mapping` for template registry
- Used `nbformat` library for notebook structure validation and I/O

### Features
- Pythonic interfaces using built-in protocols (Sequence, Mapping)
- Default templates for data analysis notebooks (intro, setup, load, explore)
- Integration support for cosmodata metadata
- Type hints throughout for type safety
- Comprehensive docstrings with examples

### API
- `Notebook`: List-like notebook manipulation
  - `append_markdown()`, `append_code()` convenience methods
  - `extend_from_template()` for applying templates
  - `from_file()` and `save()` for persistence
  - Standard sequence operations (len, indexing, slicing, iteration)
- `NotebookStore`: Dict-like file storage
  - Transparent file path management
  - Standard mapping operations (get, set, delete, contains, iteration)
- `CellTemplates`: Template registry
  - `register()` for adding custom templates
  - Standard mapping access
- Utility functions:
  - `notebook_from_metadata()`: Create from metadata dict
  - `populate_notebook()`: High-level creation with templates
  - `markdown_cell()`, `code_cell()`: Cell factories

### Design Principles
- Favor built-in protocols over custom abstractions
- Composable functions and small focused classes
- Thin wrappers over nbformat for minimal overhead
- Discoverable API using familiar Python patterns
