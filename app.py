import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
import ipywidgets
import tiledbsoma
import pyarrow as pa
import polars as pl

# --- Configuration ---
# If data folder does not exists, create one with a dummy h5ad
DATA_DIR = "data"

# --- Helper Functions ---
def get_h5ad_files():
    """Scans the DATA_DIR for .h5ad files."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".h5ad")]

def load_soma_data(filename):
    """Loads an .h5ad file and checks for UMAP coordinates."""
    try:
        soma = tiledbsoma.open(os.path.join(DATA_DIR, filename))
        var_pl = pl.from_arrow(soma.ms["RNA"].var.read().concat())
        obs_pl = pl.from_arrow(soma.obs.var.read().concat())
        return [soma, var_pl, obs_pl]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
    
# --- Accessory plots functions ---
def plot_top_genes(soma, var_pl, n_top_genes, ax, group_name=None):
    """
    Calculates the mean of each column, selects the top n columns based on mean, 
    and generates a bar plot and density plot for the selected columns.

    Args:
    soma (soma): The input soma.
    n_top_genes (int): The number of top genes to select.
    """
    X = soma.ms["RNA"].X["data"].read().tables().concat()

    gene_group_pl = pa.TableGroupBy(X, "soma_dim_1").aggregate([("soma_data", "mean")]).to_polars()

    gene_var_group_pl = gene_group_pl.join(var_pl, left_on = "soma_dim_1", right_on = "soma_joinid")
    
    gene_var_group_pl.top_k(n_top_genes, by = "soma_data_mean").plot(kind='bar', ax=ax)
        
    group_title_string = ""
    if group_name: 
        group_title_string = f"{group_name} - "

    # Bar plot
    ax.set_title(f'{group_title_string}Top {n_top_genes} Mean Gene Expressions')
    ax.set_xlabel('Genes')
    ax.set_ylabel('Mean Expression')
    ax.tick_params(axis='x',rotation=45, labelsize=10)

def create_value_counts_barplot(df, column_name, ax, group_name=None):
    """
    Generates a bar plot of value counts for a specified column in a DataFrame.

    Args:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to analyze.

    Returns:
    matplotlib.figure.Figure: The generated figure containing the bar plot.
    """

    df[column_name].value_counts().plot(kind='bar', ax=ax)  # Plot on the axes
    n_cells = df.shape[0]

    group_title_string = ""
    if group_name: 
        group_title_string = f"{group_name} - "

    ax.set_title(f"{group_title_string}Value Counts for {column_name} ({n_cells} cells)")
    ax.set_xlabel('Values')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x',rotation=45, labelsize=10)

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

            ui.input_action_button("load_data", "Load data", class_="btn-primary w-100"),
            
            ui.input_selectize("obs_cols", "2. Select up to 3 OBS columns", choices=[], multiple=True),
            
            ui.input_selectize("var_genes", "3. Select up to 3 VAR genes", choices=[], multiple=True),

            ui.input_action_button("generate_plot", "Generate Plot", class_="btn-primary w-100"),

                        ui.hr(),
            ui.h4("Group Analysis Setup"),
            
            # Group 1 Definition
            ui.h5("Group 1"),
            ui.input_select("group1_col", "Select column", choices=[], selected=None),
            ui.output_ui("group1_cat_ui"),
            
            # Group 2 Definition
            ui.h5("Group 2"),
            ui.input_select("group2_col", "Select column", choices=[], selected=None),
            ui.output_ui("group2_cat_ui"),

            # Analysis Options
            ui.h5("Analysis Options"),
            ui.input_select("analysis_col", "Column for value counts", choices=[], selected=None),

            ui.hr(),

            ui.input_action_button("analyze_selection", "Analyze Selection", class_="btn-primary w-100"),
            width=350,
        ),
        ui.output_ui("plot_or_message_ui"),
        ui.output_plot("selection_plots", height="850px"),
    )
)

# --- Shiny App Server Logic ---
def server(input, output, session):
    
    # Reactive value to hold the loaded AnnData object
    soma_reactive = reactive.Value(None)
    
    # Reactive value to hold selection choices
    obs_choices = reactive.Value([])
    var_choices = reactive.Value([])
    categorical_obs_choices = reactive.Value([])

    # reactive.Value to share the jscatter view object
    base_jscatter_view = reactive.Value(None)

    # reactive value for groups index values
    group1_indices = reactive.Value([])
    group2_indices = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.load_data)
    def _load_data():
        """Load data when a new dataset is selected."""
        notification_id = "loading_notification" 
        try:
            ui.notification_show(
                "Loading data... Please wait.",
                id=notification_id,
                duration=None,
                close_button=False,
                type="message" # Other types: "warning", "error", "default"
            )

            dataset_file = input.dataset()
            if dataset_file:
                print(f"Loading dataset: {dataset_file}")
                soma, var, obs = load_adata(dataset_file)
                print(f"Dataset loaded")
                if soma is not None:
                    soma_reactive.set(soma)
                    # Update choices for obs and var selectors
                    obs_cols = [col for col in obs.columns]
                    var_genes = list(var[["var_id"]])
                    cat_cols = list(obs.select_dtypes(include=['category', 'object']).columns)
                    
                    obs_choices.set(obs_cols)
                    var_choices.set(var_genes)
                    categorical_obs_choices.set(cat_cols)

        except Exception as e:
            print(f"An error occurred: {e}")
            ui.notification_show(
                f"Error loading data: {e}",
                duration=10,
                type="error"
            )

        finally:
            # Remove the loading notification
            ui.notification_remove(notification_id)

    @reactive.Effect
    def _update_selectors():
        """Update the UI selectors with new choices."""
        ui.update_selectize("obs_cols", choices=obs_choices.get(), selected=[])
        ui.update_selectize("var_genes", choices=var_choices.get(), selected=[])
        
        cat_cols = categorical_obs_choices.get()
        ui.update_select("group1_col", choices=cat_cols, selected=None if not cat_cols else cat_cols[0])
        ui.update_select("group2_col", choices=cat_cols, selected=None if not cat_cols else cat_cols[0])
        ui.update_select("analysis_col", choices=cat_cols, selected=None if not cat_cols else cat_cols[0])

    @render.ui
    def group1_cat_ui():
        obs = obs_choices.get()
        col = input.group1_col()
        if obs is None or not col:
            return None
        categories = list(obs[col].astype('category').cat.categories)
        return ui.input_selectize("group1_cat", "Select categories for Group 1", choices=categories, multiple=True)

    @render.ui
    def group2_cat_ui():
        obs = obs_choices.get()
        col = input.group2_col()
        if obs is None or not col:
            return None
        categories = list(obs[col].astype('category').cat.categories)
        return ui.input_selectize("group2_cat", "Select categories for Group 2", choices=categories, multiple=True)

    @render.ui
    @reactive.event(input.generate_plot, input.analyze_selection)
    def plot_or_message_ui():
        # Check we have the data or print error messages
        soma = soma_reactive.get()
        if not input.generate_plot():
            return ui.p("Please select a dataset and options, then click 'Generate Plot'.")
        if soma is None:
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
        
        soma = soma_reactive.get()
        obs = obs_choices.get()
        selected_obs = list(input.obs_cols())
        selected_vars = list(input.var_genes())

        group1_selection = group1_indices.get()
        group2_selection = group2_indices.get()
        
        UMAP = soma.ms["RNA"].obsm['X_umap'].read().tables().concat().to_pandas().pivot(
        index="soma_dim_0",  # Cell/observation index
        columns="soma_dim_1",  # UMAP dimension (0=UMAP_1, 1=UMAP_2)
        values="soma_data"  # Coordinate values
        ).rename(columns={0: "UMAP_1", 1: "UMAP_2"})

        plot_df = UMAP
        
        if selected_obs:
            plot_df = plot_df.join(obs[selected_obs])
            
        if selected_vars:
            gene_expression = soma[:, selected_vars].to_df()
            plot_df = plot_df.join(gene_expression)
        
        try:
            views = []
            base_view = jscatter.Scatter(x='UMAP1', y='UMAP2', color_by=selected_obs[0], data=plot_df, legend=True, data_use_index=True)
                        
            # compose the linked view using jscatter
            views.append(base_view)
            for col in selected_obs[1:] + selected_vars:
                p = jscatter.Scatter(x='UMAP1', y='UMAP2', color_by=col, data=plot_df, legend=True)
                p = p.tooltip(True, properties=[col])
                views.append(p)

            if len(group1_selection) > 0 and len(group2_selection) > 0:
                plot_df['app_grouping'] = 'NotSelected'
                plot_df.loc[group1_selection, 'app_grouping'] = 'Group1'
                plot_df.loc[group2_selection, 'app_grouping'] = 'Group2'
                p = jscatter.Scatter(x='UMAP1', y='UMAP2', 
                                     color_by="app_grouping", 
                                     color_map=dict(
                                        Group1="#29876E", Group2="#5188E2", NotSelected="#E6E6EF"
                                    ), 
                                    data=plot_df, legend=True)
                p = p.tooltip(True, properties=[input.analysis_col()])
                views.append(p)

            composed_view = jscatter.compose(views, sync_view=True, sync_hover=True, sync_selection=True, cols=2)
            base_jscatter_view.set(base_view)

            return composed_view
        except Exception as e:
            # Can't return a ui.p, so return None to render nothing in case of exception
            print(f"Error during jscatter creation: {e}")
            return None
    
    @render.plot() 
    @reactive.event(input.analyze_selection)    
    def selection_plots():
        base_view = base_jscatter_view.get()
        soma = soma_reactive.get()
        obs = obs_choices.get()

        g1_col, g1_cat = input.group1_col(), input.group1_cat()
        g2_col, g2_cat = input.group2_col(), input.group2_cat()
        analysis_col = input.analysis_col()

        # Validate that all required inputs are present
        if not all([soma is not None, g1_col, g1_cat, g2_col, g2_cat, analysis_col]):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Please define both groups and select an analysis column.", ha='center', va='center')
            ax.axis('off')
            return fig

        try:
            indices1 = obs.index[obs[g1_col].isin(g1_cat)]
            indices2 = obs.index[obs[g2_col].isin(g2_cat)]
            group1_indices.set(indices1)
            group2_indices.set(indices2)
            print(f"Group 1: {len(indices1)} cells, Group 2: {len(indices2)} cells")
            
            if len(indices1) == 0 or len(indices2) == 0:
                fig, ax = plt.subplots();
                ax.text(0.5, 0.5, "One or both groups have 0 cells. Check selections.", ha='center', va='center')
                ax.axis('off')
                return fig
        except Exception as e:
            print(f"Error getting indices: {e}")
            return None

        selected_cells = indices1.union(indices2).tolist()
        # print(f"Selected cells: {selected_cells[1:5]}")
        base_view.selection(selected_cells)

        adata_g1 = soma[indices1, :] #.copy()
        adata_g2 = soma[indices2, :] #.copy()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
        fig.suptitle("Paired Group Analysis", fontsize=18)
        n_top_genes_plot = 5

        create_value_counts_barplot(adata_g1.obs, analysis_col, axes[0, 0], "Group 1")
        plot_top_genes(adata_g1, n_top_genes_plot, axes[0, 1], "Group 1")
        create_value_counts_barplot(adata_g2.obs, analysis_col, axes[1, 0], "Group 2")
        plot_top_genes(adata_g2, n_top_genes_plot, axes[1, 1], "Group 2")        

        return fig

# Create the Shiny app instance
app = App(app_ui, server)

# To run this app:
# Run `shiny run --reload app.py` in your terminal.
