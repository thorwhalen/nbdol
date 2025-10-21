"""End-to-end demonstration of nbdol workflow.

This script demonstrates a complete workflow for generating and managing
analysis notebooks for multiple datasets using nbdol with cosmodata.
"""

from pathlib import Path
from nbdol import Notebook, NotebookStore, CellTemplates, populate_notebook
import nbformat


# ============================================================================
# Step 1: Simulate cosmodata metadata
# ============================================================================

def create_sample_metadata():
    """Create sample metadata simulating cosmodata.metas."""
    return {
        'bitcoin': {
            'title': 'Bitcoin Trading Data',
            'description': 'Historical Bitcoin prices, volumes, and market metrics',
            'src': 'https://example.com/data/bitcoin.parquet',
            'target_filename': 'bitcoin.parquet',
            'ext': None,
            'viz_columns_info': 'Key columns: timestamp, price, volume, market_cap',
            'peep_mode': 'short',
            'peep_exclude_cols': ['internal_id']
        },
        'weather': {
            'title': 'Global Weather Patterns',
            'description': 'Daily temperature, precipitation, and weather conditions',
            'src': 'https://example.com/data/weather.csv',
            'target_filename': 'weather.csv',
            'ext': 'csv',
            'viz_columns_info': 'Key columns: date, location, temp, precipitation',
            'peep_mode': 'sample',
            'peep_exclude_cols': []
        },
        'covid': {
            'title': 'COVID-19 Statistics',
            'description': 'Global COVID-19 cases, deaths, and vaccination rates',
            'src': 'https://example.com/data/covid.parquet',
            'target_filename': 'covid.parquet',
            'ext': None,
            'viz_columns_info': 'Key columns: date, country, cases, deaths, vaccinations',
            'peep_mode': 'stats',
            'peep_exclude_cols': ['region_code']
        }
    }


# ============================================================================
# Step 2: Generate notebooks for all datasets
# ============================================================================

def generate_all_notebooks(metas, output_dir='generated_notebooks/'):
    """Generate notebooks for all datasets in metadata.
    
    Args:
        metas: Dictionary of dataset metadata
        output_dir: Directory to save notebooks
    
    Returns:
        NotebookStore with all generated notebooks
    """
    print("=" * 70)
    print("GENERATING NOTEBOOKS")
    print("=" * 70)
    
    store = NotebookStore(output_dir)
    
    for dataset_key, metadata in metas.items():
        print(f"\nGenerating notebook for: {dataset_key}")
        print(f"  Title: {metadata['title']}")
        
        # Generate notebook with default templates
        nb = populate_notebook(
            metadata,
            template_sequence=['intro', 'setup', 'load', 'explore'],
            n_viz_cells=5
        )
        
        # Add dataset-specific sections
        _add_custom_sections(nb, dataset_key)
        
        # Save to store
        store[dataset_key] = nb
        print(f"  ✓ Saved to: {output_dir}{dataset_key}.ipynb")
    
    print(f"\n✓ Generated {len(store)} notebooks")
    return store


def _add_custom_sections(nb, dataset_key):
    """Add dataset-specific analysis sections."""
    if dataset_key == 'bitcoin':
        nb.append_markdown("## Price Analysis")
        nb.append_code("""# Analyze price trends
price_trend = data['price'].rolling(window=7).mean()
price_trend.plot(title='7-day Moving Average')""")
        
    elif dataset_key == 'weather':
        nb.append_markdown("## Temperature Distribution")
        nb.append_code("""# Analyze temperature patterns
data['temp'].hist(bins=50)
data.groupby('location')['temp'].mean().plot(kind='bar')""")
        
    elif dataset_key == 'covid':
        nb.append_markdown("## Case Growth Analysis")
        nb.append_code("""# Analyze case growth rates
data['daily_cases'] = data.groupby('country')['cases'].diff()
top_countries = data.groupby('country')['cases'].max().nlargest(10)
print(top_countries)""")


# ============================================================================
# Step 3: Batch update existing notebooks
# ============================================================================

def add_performance_metrics_section(store):
    """Add performance metrics section to all notebooks.
    
    Args:
        store: NotebookStore instance
    """
    print("\n" + "=" * 70)
    print("ADDING PERFORMANCE METRICS SECTION")
    print("=" * 70)
    
    for key in store:
        print(f"  Updating {key}.ipynb...")
        
        nb = store[key]
        
        # Add performance tracking section
        nb.append_markdown("## Performance Metrics")
        nb.append_code("""import time

# Track execution time
start_time = time.time()

# ... your analysis code ...

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
""")
        
        # Save changes
        store[key] = nb
    
    print(f"\n✓ Updated {len(store)} notebooks")


# ============================================================================
# Step 4: Create custom template for specific use case
# ============================================================================

def create_report_notebooks(metas, output_dir='report_notebooks/'):
    """Create executive report notebooks with custom template.
    
    Args:
        metas: Dictionary of dataset metadata
        output_dir: Directory to save notebooks
    
    Returns:
        NotebookStore with report notebooks
    """
    print("\n" + "=" * 70)
    print("GENERATING REPORT NOTEBOOKS WITH CUSTOM TEMPLATE")
    print("=" * 70)
    
    # Create custom templates
    templates = CellTemplates()
    
    def report_intro_template(meta):
        """Executive report intro template."""
        yield nbformat.v4.new_markdown_cell(f"# Executive Report: {meta['title']}")
        yield nbformat.v4.new_markdown_cell("---")
        yield nbformat.v4.new_markdown_cell(
            f"**Dataset:** {meta['title']}\n\n"
            f"**Description:** {meta['description']}\n\n"
            f"**Generated:** 2024-01-19"
        )
        yield nbformat.v4.new_markdown_cell("## Executive Summary")
        yield nbformat.v4.new_code_cell(
            "# Key metrics will be displayed here\n"
            "print('Key Findings:')\n"
            "print('- Metric 1: ...')\n"
            "print('- Metric 2: ...')"
        )
    
    def report_visualizations_template(meta):
        """Report visualizations template."""
        yield nbformat.v4.new_markdown_cell("## Key Visualizations")
        yield nbformat.v4.new_code_cell(
            "# Create summary visualizations\n"
            "import matplotlib.pyplot as plt\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
            "# Plot 1: Trend\n"
            "# Plot 2: Distribution\n"
            "# Plot 3: Comparison\n"
            "# Plot 4: Forecast"
        )
    
    templates.register('report_intro', report_intro_template)
    templates.register('report_viz', report_visualizations_template)
    
    # Generate report notebooks
    store = NotebookStore(output_dir)
    
    for key, metadata in metas.items():
        print(f"  Generating report for: {key}")
        
        # Use custom template sequence
        nb = Notebook()
        nb.extend_from_template(templates['report_intro'], metadata)
        nb.extend_from_template(templates['report_viz'], metadata)
        
        nb.append_markdown("## Recommendations")
        nb.append_code("# Analysis-based recommendations\nprint('TODO: Add recommendations')")
        
        store[f"{key}_report"] = nb
    
    print(f"\n✓ Generated {len(store)} report notebooks")
    return store


# ============================================================================
# Step 5: Demonstrate notebook modification
# ============================================================================

def modify_specific_notebook(store, notebook_key):
    """Demonstrate modifying a specific notebook.
    
    Args:
        store: NotebookStore instance
        notebook_key: Key of notebook to modify
    """
    print("\n" + "=" * 70)
    print(f"MODIFYING NOTEBOOK: {notebook_key}")
    print("=" * 70)
    
    if notebook_key not in store:
        print(f"  ✗ Notebook '{notebook_key}' not found")
        return
    
    # Load notebook
    nb = store[notebook_key]
    print(f"  Original cells: {len(nb)}")
    
    # Modify title
    if nb[0]['cell_type'] == 'markdown':
        original_title = nb[0]['source']
        nb[0]['source'] = nb[0]['source'].replace('#', '###')
        print(f"  Modified title: {original_title[:30]}...")
    
    # Insert new cell at position 2
    nb.insert(2, nbformat.v4.new_markdown_cell("## Important Note\n\nThis is a modified notebook."))
    
    # Append summary section
    nb.append_markdown("## Summary")
    nb.append_code("# Summarize findings\nprint('Analysis complete!')")
    
    print(f"  Final cells: {len(nb)}")
    
    # Save changes
    store[notebook_key] = nb
    print(f"  ✓ Saved modifications")


# ============================================================================
# Step 6: Demonstrate sequence operations
# ============================================================================

def demonstrate_sequence_ops(store, notebook_key):
    """Demonstrate sequence protocol operations on notebook.
    
    Args:
        store: NotebookStore instance
        notebook_key: Key of notebook to analyze
    """
    print("\n" + "=" * 70)
    print("SEQUENCE OPERATIONS DEMO")
    print("=" * 70)
    
    nb = store[notebook_key]
    
    print(f"\nNotebook: {notebook_key}")
    print(f"  Total cells: {len(nb)}")
    
    # Count cell types
    markdown_count = sum(1 for cell in nb if cell['cell_type'] == 'markdown')
    code_count = sum(1 for cell in nb if cell['cell_type'] == 'code')
    
    print(f"  Markdown cells: {markdown_count}")
    print(f"  Code cells: {code_count}")
    
    # Display first 3 cells
    print("\n  First 3 cells:")
    for i, cell in enumerate(nb[:3]):
        source_preview = cell['source'][:50].replace('\n', ' ')
        print(f"    [{i}] {cell['cell_type']}: {source_preview}...")
    
    # Filter cells
    markdown_cells = [cell for cell in nb if cell['cell_type'] == 'markdown']
    print(f"\n  Markdown cell indices: {[i for i, c in enumerate(nb) if c['cell_type'] == 'markdown']}")


# ============================================================================
# Main workflow
# ============================================================================

def main():
    """Run complete demonstration workflow."""
    # Step 1: Create sample metadata
    metas = create_sample_metadata()
    
    # Step 2: Generate all notebooks
    analysis_store = generate_all_notebooks(metas, 'analysis_notebooks/')
    
    # Step 3: Batch update
    add_performance_metrics_section(analysis_store)
    
    # Step 4: Create custom reports
    report_store = create_report_notebooks(metas, 'report_notebooks/')
    
    # Step 5: Modify specific notebook
    modify_specific_notebook(analysis_store, 'bitcoin')
    
    # Step 6: Demonstrate sequence operations
    demonstrate_sequence_ops(analysis_store, 'weather')
    
    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"\nGenerated:")
    print(f"  - {len(analysis_store)} analysis notebooks in analysis_notebooks/")
    print(f"  - {len(report_store)} report notebooks in report_notebooks/")
    print(f"\nAll notebooks can be managed using:")
    print(f"  - store['key'] to load")
    print(f"  - store['key'] = nb to save")
    print(f"  - list(store) to list all")
    print(f"  - 'key' in store to check existence")
    print(f"  - del store['key'] to delete")


if __name__ == '__main__':
    main()
