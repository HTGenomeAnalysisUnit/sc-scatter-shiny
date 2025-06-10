import os
import pandas as pd
import scanpy as sc
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

# --- Configuration ---
# If data folder does not exists, create one with a dummy h5ad
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

    import numpy as np
    import anndata
    
    n_obs, n_vars = 1000, 500
    adata = anndata.AnnData(np.random.randn(n_obs, n_vars))
    adata.obs['cell_type'] = np.random.choice(['T-cell', 'B-cell', 'Macrophage'], n_obs)
    adata.obs['leiden'] = np.random.choice([f'cluster_{i}' for i in range(5)], n_obs)
    adata.obs['total_counts'] = np.random.randint(1000, 5000, n_obs)
    adata.var_names = [f'Gene_{i}' for i in range(n_vars)]
    # Generate dummy UMAP coordinates
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    
    adata.write_h5ad(os.path.join(DATA_DIR, 'sample_data.h5ad'))

# --- Helper Functions ---
def get_h5ad_files():
    """Scans the DATA_DIR for .h5ad files."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".h5ad")]

def load_adata(filename):
    """Loads an .h5ad file and checks for UMAP coordinates."""
    try:
        adata = sc.read_h5ad(os.path.join(DATA_DIR, filename))
        if 'X_umap' not in adata.obsm:
            # Make UMAP if not pre-computed, just for example data
            sc.pp.pca(adata)
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)
        return adata
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# --- Shiny App UI ---
app_ui = ui.page_fluid(
    ui.tags.style("""
        .shiny-input-container { margin-bottom: 20px; }
        .btn-primary { background-color: #007bff; border-color: #007bff; }
        h2 { color: #333; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    """),
    ui.h2("Interactive Single-Cell UMAP Explorer"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_selectize("dataset", "1. Select Dataset", choices=get_h5ad_files(), selected=None),
            
            ui.input_selectize("obs_cols", "2. Select up to 3 OBS columns", choices=[], multiple=True),
            
            ui.input_selectize("var_genes", "3. Select up to 3 VAR genes", choices=[], multiple=True),

            ui.input_action_button("generate_plot", "Generate Plot", class_="btn-primary w-100"),
            width=350,
        ),
        ui.output_ui("plot_or_message_ui"),
    )
)

# --- Shiny App Server Logic ---
def server(input, output, session):

    # Reactive value to hold the loaded AnnData object
    adata_reactive = reactive.Value(None)
    
    # Reactive value to hold selection choices
    obs_choices = reactive.Value([])
    var_choices = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.dataset)
    def _load_data():
        """Load data when a new dataset is selected."""
        dataset_file = input.dataset()
        if dataset_file:
            adata = load_adata(dataset_file)
            if adata is not None:
                adata_reactive.set(adata)
                # Update choices for obs and var selectors
                obs_cols = [col for col in adata.obs.columns]
                var_genes = list(adata.var_names)
                
                obs_choices.set(obs_cols)
                var_choices.set(var_genes)

    @reactive.Effect
    def _update_selectors():
        """Update the UI selectors with new choices."""
        ui.update_selectize("obs_cols", choices=obs_choices.get(), selected=[])
        ui.update_selectize("var_genes", choices=var_choices.get(), selected=[])

    @render.ui
    @reactive.event(input.generate_plot)
    def plot_or_message_ui():
        # This function runs when the button is clicked.
        # It performs all checks.
        adata = adata_reactive.get()
        if not input.generate_plot():
            return ui.p("Please select a dataset and options, then click 'Generate Plot'.")
        if adata is None:
            return ui.p("Please select a valid dataset.")
            
        selected_obs = input.obs_cols()
        selected_vars = input.var_genes()
        
        if not selected_obs and not selected_vars:
            return ui.p("Please select at least one OBS column or VAR gene to color by.")

        if len(selected_obs) > 3 or len(selected_vars) > 3:
            return ui.p("Please select a maximum of 3 OBS columns and 3 VAR genes.")
        
        # If all checks pass, return the output_widget.
        return output_widget("scatter_plot_output")

    # Generate the jscatter plot
    @render_widget
    def scatter_plot_output():
        import jscatter
        
        adata = adata_reactive.get()
        selected_obs = list(input.obs_cols())
        selected_vars = list(input.var_genes())
        
        umap_coords = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'], index=adata.obs.index)
        plot_df = umap_coords.copy()
        
        if selected_obs:
            plot_df = plot_df.join(adata.obs[selected_obs])
            
        if selected_vars:
            gene_expression = adata[:, selected_vars].to_df()
            plot_df = plot_df.join(gene_expression)
        
        try:
            views = []
            for col in selected_obs + selected_vars:
                p = jscatter.Scatter(x='UMAP1', y='UMAP2', color_by=col, data=plot_df, legend=True)
                views.append(p)
            return jscatter.compose(views, sync_view=True, sync_hover=True, sync_selection=True, rows=2)
        except Exception as e:
            # Can't return a ui.p, so return None to render nothing in case of exception
            print(f"Error during jscatter creation: {e}")
            return None

# Create the Shiny app instance
app = App(app_ui, server)

# To run this app:
# Run `shiny run --reload app.py` in your terminal.
