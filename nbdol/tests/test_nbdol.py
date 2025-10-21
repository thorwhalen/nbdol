"""Tests for nbdol base classes.

Run with: pytest nbdol/tests/
"""

import pytest
import tempfile
from pathlib import Path
import nbformat

from nbdol.base import Notebook, NotebookStore, CellTemplates, markdown_cell, code_cell


class TestNotebook:
    """Tests for Notebook class."""

    def test_init_empty(self):
        """Test creating empty notebook."""
        nb = Notebook()
        assert len(nb) == 0
        assert isinstance(nb._nb, nbformat.NotebookNode)

    def test_append_markdown(self):
        """Test appending markdown cell."""
        nb = Notebook()
        nb.append_markdown("# Title")

        assert len(nb) == 1
        assert nb[0]['cell_type'] == 'markdown'
        assert nb[0]['source'] == "# Title"

    def test_append_code(self):
        """Test appending code cell."""
        nb = Notebook()
        nb.append_code("x = 42")

        assert len(nb) == 1
        assert nb[0]['cell_type'] == 'code'
        assert nb[0]['source'] == "x = 42"

    def test_sequence_operations(self):
        """Test list-like operations."""
        nb = Notebook()

        # Append multiple cells
        nb.append_markdown("# Title")
        nb.append_code("x = 1")
        nb.append_markdown("## Section")

        # Length
        assert len(nb) == 3

        # Indexing
        assert nb[0]['cell_type'] == 'markdown'
        assert nb[1]['cell_type'] == 'code'

        # Slicing
        first_two = nb[:2]
        assert len(first_two) == 2

        # Iteration
        cell_types = [cell['cell_type'] for cell in nb]
        assert cell_types == ['markdown', 'code', 'markdown']

    def test_insert(self):
        """Test inserting cell at position."""
        nb = Notebook()
        nb.append_markdown("# First")
        nb.append_markdown("# Third")

        nb.insert(1, markdown_cell("# Second"))

        assert len(nb) == 3
        assert nb[1]['source'] == "# Second"

    def test_delete(self):
        """Test deleting cell."""
        nb = Notebook()
        nb.append_markdown("# First")
        nb.append_markdown("# Second")
        nb.append_markdown("# Third")

        del nb[1]

        assert len(nb) == 2
        assert nb[1]['source'] == "# Third"

    def test_setitem(self):
        """Test modifying cell."""
        nb = Notebook()
        nb.append_markdown("# Original")

        nb[0] = markdown_cell("# Modified")

        assert nb[0]['source'] == "# Modified"

    def test_extend_from_template(self):
        """Test extending with template function."""
        nb = Notebook()

        def test_template(meta):
            yield markdown_cell(f"# {meta['title']}")
            yield code_cell("import pandas")

        nb.extend_from_template(test_template, {'title': 'Test'})

        assert len(nb) == 2
        assert nb[0]['source'] == "# Test"
        assert nb[1]['source'] == "import pandas"

    def test_save_and_load(self):
        """Test saving and loading notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.ipynb'

            # Create and save
            nb1 = Notebook()
            nb1.append_markdown("# Test")
            nb1.append_code("x = 42")
            nb1.save(path)

            # Load
            nb2 = Notebook.from_file(path)

            assert len(nb2) == 2
            assert nb2[0]['source'] == "# Test"
            assert nb2[1]['source'] == "x = 42"

    def test_to_dict(self):
        """Test exporting to dict."""
        nb = Notebook()
        nb.append_markdown("# Test")

        d = nb.to_dict()

        assert d['nbformat'] == 4
        assert len(d['cells']) == 1
        # When converted to dict, source becomes a list of lines
        cell_source = d['cells'][0]['source']
        if isinstance(cell_source, list):
            assert cell_source == ['# Test']
        else:
            assert cell_source == "# Test"


class TestNotebookStore:
    """Tests for NotebookStore class."""

    def test_init(self):
        """Test creating store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)
            assert store._root == Path(tmpdir)

    def test_setitem_getitem(self):
        """Test saving and loading notebooks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)

            # Create notebook
            nb1 = Notebook()
            nb1.append_markdown("# Test")

            # Save via store
            store['test'] = nb1

            # Load via store
            nb2 = store['test']

            assert len(nb2) == 1
            assert nb2[0]['source'] == "# Test"

    def test_delitem(self):
        """Test deleting notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)

            # Create and save
            nb = Notebook()
            store['test'] = nb

            # Delete
            del store['test']

            # Verify deleted
            assert 'test' not in store
            assert not (Path(tmpdir) / 'test.ipynb').exists()

    def test_contains(self):
        """Test checking notebook existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)

            nb = Notebook()
            store['exists'] = nb

            assert 'exists' in store
            assert 'missing' not in store

    def test_iter(self):
        """Test iterating over notebook keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)

            # Create multiple notebooks
            for name in ['nb1', 'nb2', 'nb3']:
                store[name] = Notebook()

            keys = list(store)
            assert len(keys) == 3
            assert set(keys) == {'nb1', 'nb2', 'nb3'}

    def test_len(self):
        """Test getting notebook count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)

            assert len(store) == 0

            store['nb1'] = Notebook()
            assert len(store) == 1

            store['nb2'] = Notebook()
            assert len(store) == 2

    def test_keyerror_on_missing(self):
        """Test KeyError raised for missing notebook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = NotebookStore(tmpdir)

            with pytest.raises(KeyError):
                _ = store['missing']

            with pytest.raises(KeyError):
                del store['missing']


class TestCellTemplates:
    """Tests for CellTemplates class."""

    def test_register_and_getitem(self):
        """Test registering and retrieving templates."""
        templates = CellTemplates()

        def my_template(meta):
            yield markdown_cell("test")

        templates.register('test', my_template)

        retrieved = templates['test']
        assert retrieved == my_template

    def test_iter(self):
        """Test iterating over template names."""
        templates = CellTemplates()

        templates.register('t1', lambda m: [])
        templates.register('t2', lambda m: [])

        names = list(templates)
        assert set(names) == {'t1', 't2'}

    def test_len(self):
        """Test getting template count."""
        templates = CellTemplates()

        assert len(templates) == 0

        templates.register('t1', lambda m: [])
        assert len(templates) == 1

        templates.register('t2', lambda m: [])
        assert len(templates) == 2

    def test_keyerror_on_missing(self):
        """Test KeyError for missing template."""
        templates = CellTemplates()

        with pytest.raises(KeyError):
            _ = templates['missing']


class TestUtilFunctions:
    """Tests for utility functions."""

    def test_markdown_cell(self):
        """Test creating markdown cell."""
        cell = markdown_cell("# Title")

        assert cell['cell_type'] == 'markdown'
        assert cell['source'] == "# Title"

    def test_code_cell(self):
        """Test creating code cell."""
        cell = code_cell("x = 42")

        assert cell['cell_type'] == 'code'
        assert cell['source'] == "x = 42"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
