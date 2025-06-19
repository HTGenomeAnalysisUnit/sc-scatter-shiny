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
DATA_DIR = "data/"

# --- Helper Functions ---
def get_soma_files():
    """Scans the DATA_DIR for soma files."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith("_soma")]

def load_soma_data(filename):
    """Loads an soma file and checks for UMAP coordinates."""
    try:
        soma = tiledbsoma.open(os.path.join(DATA_DIR, filename))
        # Check for UMAP coordinates
        if "X_umap" not in soma.ms["RNA"].obsm:
            print(f"Warning: UMAP coordinates not found in {filename}")
            return None
        return soma
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
    
# --- Accessory plots functions ---
def plot_top_genes(soma, n_top_genes, ax, group_name=None):
    """
    Calculates the mean of each column, selects the top n columns based on mean, 
    and generates a bar plot for the selected columns.

    Args:
    soma (soma): The input soma.
    n_top_genes (int): The number of top genes to select.
    """
    # This function operates on a SOMA slice, so reading should be efficient.
    X = soma.ms["RNA"].X["data"].read().tables().concat()
    var_pl = pl.from_arrow(soma.ms["RNA"].var.read().concat())
    
    # Using Polars for efficient aggregation
    gene_group_pl = pa.TableGroupBy(X, "soma_dim_1").aggregate([("soma_data", "mean")]).to_polars()
    gene_var_group_pl = gene_group_pl.join(var_pl, left_on = "soma_dim_1", right_on = "soma_joinid")
    
    # Select top genes and convert to pandas for plotting
    top_genes_df = gene_var_group_pl.top_k(n_top_genes, by = "soma_data_mean").to_pandas()

    group_title_string = ""
    if group_name:
        group_title_string = f"{group_name} - "

    # Bar plot using pandas on the specified axes
    top_genes_df.set_index('var_id')['soma_data_mean'].plot(kind='bar', ax=ax)
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
    """
    if df is None or column_name not in df.columns:
        ax.text(0.5, 0.5, f"Column '{column_name}' not found.", ha='center', va='center')
        ax.axis('off')
        return

    df[column_name].value_counts().plot(kind='bar', ax=ax)
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
            ui.input_selectize("dataset", "1. Select Dataset", choices=get_soma_files(), selected=None),
            ui.input_action_button("load_data", "Load data", class_="btn-primary w-100"),
            ui.input_selectize("obs_cols", "2. Select up to 3 OBS columns", choices=[], multiple=True),
            ui.input_selectize("var_genes", "3. Select up to 3 VAR genes", choices=[], multiple=True),
            ui.input_action_button("generate_plot", "Generate Plot", class_="btn-primary w-100"),
            ui.hr(),
            ui.h4("Group Analysis Setup"),
            ui.h5("Group 1"),
            ui.input_select("group1_col", "Select column", choices=[], selected=None),
            ui.output_ui("group1_cat_ui"),
            ui.h5("Group 2"),
            ui.input_select("group2_col", "Select column", choices=[], selected=None),
            ui.output_ui("group2_cat_ui"),
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
    
    # Reactive values to hold the loaded data and state
    soma_reactive = reactive.Value(None)
    obs_df_reactive = reactive.Value(None)
    var_df_reactive = reactive.Value(None)
    
    # Reactive values for selector choices
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
        dataset_file = input.dataset()
        if not dataset_file:
            ui.notification_show("Please select a dataset.", duration=5, type="warning")
            return

        notification_id = "loading_notification" 
        ui.notification_show(
            "Loading data... Please wait.",
            id=notification_id,
            duration=None,
            close_button=False,
            type="message"
        )
        try:
            print(f"Loading dataset: {dataset_file}")
            soma = load_soma_data(dataset_file)
            print(f"Dataset loaded: {soma}")
            if soma is not None:
                soma_reactive.set(soma)
                
                # Read obs and var data into pandas once and store in reactive values
                obs_pd = soma.obs.read().concat().to_pandas()
                var_pd = soma.ms["RNA"].var.read().concat().to_pandas()
                obs_df_reactive.set(obs_pd)
                var_df_reactive.set(var_pd)

                # Update choices for obs and var selectors
                obs_cols = obs_pd.columns.tolist()
                var_genes = var_pd["var_id"].tolist()
                cat_cols = obs_pd.select_dtypes(include=['category', 'object']).columns.tolist()
                
                obs_choices.set(obs_cols)
                var_choices.set(var_genes)
                categorical_obs_choices.set(cat_cols)
                ui.notification_show("Data loaded successfully!", duration=5, type="message")
            else:
                ui.notification_show(f"Failed to load {dataset_file}. It might be invalid or lack UMAP data.", duration=10, type="error")

        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            ui.notification_show(f"Error loading data: {e}", duration=10, type="error")
        finally:
            ui.notification_remove(notification_id)

    @reactive.Effect
    def _update_selectors():
        """Update the UI selectors with new choices."""
        obs_cols = obs_choices.get()
        var_genes = var_choices.get()
        cat_cols = categorical_obs_choices.get()

        ui.update_selectize("obs_cols", choices=obs_cols, selected=[])
        ui.update_selectize("var_genes", choices=var_genes, selected=[])
        
        selected_cat = cat_cols[0] if cat_cols else None
        ui.update_select("group1_col", choices=cat_cols, selected=selected_cat)
        ui.update_select("group2_col", choices=cat_cols, selected=selected_cat)
        ui.update_select("analysis_col", choices=cat_cols, selected=selected_cat)

    def _render_group_cat_ui(col_input):
        obs_pd = obs_df_reactive.get()
        col = col_input()
        if obs_pd is None or not col:
            return None
        categories = obs_pd[col].astype('category').cat.categories.tolist()
        return categories

    @render.ui
    def group1_cat_ui():
        categories = _render_group_cat_ui(input.group1_col)
        return ui.input_selectize("group1_cat", "Select categories for Group 1", choices=categories or [], multiple=True)

    @render.ui
    def group2_cat_ui():
        categories = _render_group_cat_ui(input.group2_col)
        return ui.input_selectize("group2_cat", "Select categories for Group 2", choices=categories or [], multiple=True)

    @render.ui
    @reactive.event(input.generate_plot, input.analyze_selection)
    def plot_or_message_ui():
        if input.generate_plot() == 0 and input.analyze_selection() == 0:
            return ui.p("Please select a dataset and options, then click 'Generate Plot'.")
        if soma_reactive.get() is None:
            return ui.p("Please load a valid dataset first.")
            
        selected_obs = input.obs_cols()
        selected_vars = input.var_genes()
        
        if not selected_obs and not selected_vars:
            return ui.p("Please select at least one OBS column or VAR gene to color by.")

        if len(selected_obs) > 3 or len(selected_vars) > 3:
            return ui.p("Please select a maximum of 3 OBS columns and 3 VAR genes.")
        
        return output_widget("scatter_plot_output")

    @render_widget
    def scatter_plot_output():
        import jscatter
        
        soma = soma_reactive.get()
        obs_df = obs_df_reactive.get()
        var_df = var_df_reactive.get()
        selected_obs = list(input.obs_cols())
        selected_vars = list(input.var_genes())

        group1_selection = group1_indices.get()
        group2_selection = group2_indices.get()
        
        try:
            UMAP = soma.ms["RNA"].obsm['X_umap'].read().tables().concat().to_pandas().pivot(
                index="soma_dim_0",
                columns="soma_dim_1",
                values="soma_data"
            ).rename(columns={0: "UMAP_1", 1: "UMAP_2"})

            plot_df = UMAP
            
            if selected_obs:
                plot_df = plot_df.join(obs_df[selected_obs])
                
            if selected_vars:
                var_mask = var_df['var_id'].isin(selected_vars)
                var_indices = var_df.index[var_mask]
                
                gene_expression_long = soma.ms['RNA'].X['data'].read(coords=(slice(None), var_indices)).to_pandas()
                gene_expression_wide = gene_expression_long.pivot(index='soma_dim_0', columns='soma_dim_1', values='soma_data')
                
                # Map var indices to var IDs (gene names) for column headers
                gene_names = var_df.loc[var_indices, 'var_id']
                gene_expression_wide.columns = gene_names
                
                plot_df = plot_df.join(gene_expression_wide)

            views = []
            
            # Determine the primary coloring column
            color_by_col = None
            if selected_obs:
                color_by_col = selected_obs[0]
            elif selected_vars:
                color_by_col = selected_vars[0]

            base_view = jscatter.Scatter(x='UMAP_1', y='UMAP_2', color_by=color_by_col, data=plot_df, legend=True, data_use_index=True)
            views.append(base_view)
            
            all_color_by_cols = selected_obs[1:] + selected_vars
            if not selected_obs and len(selected_vars) > 1:
                all_color_by_cols = selected_vars[1:]

            for col in all_color_by_cols:
                p = jscatter.Scatter(x='UMAP_1', y='UMAP_2', color_by=col, data=plot_df, legend=True)
                p = p.tooltip(True, properties=[col])
                views.append(p)

            if len(group1_selection) > 0 and len(group2_selection) > 0:
                plot_df['app_grouping'] = 'NotSelected'
                plot_df.loc[group1_selection, 'app_grouping'] = 'Group1'
                plot_df.loc[group2_selection, 'app_grouping'] = 'Group2'
                p = jscatter.Scatter(x='UMAP_1', y='UMAP_2', 
                                     color_by="app_grouping", 
                                     color_map={'Group1': "#29876E", 'Group2': "#5188E2", 'NotSelected': "#E6E6EF"},
                                     data=plot_df, legend=True)
                p = p.tooltip(True, properties=[input.analysis_col()])
                views.append(p)

            composed_view = jscatter.compose(views, sync_view=True, sync_hover=True, sync_selection=True, cols=2)
            base_jscatter_view.set(base_view)

            return composed_view
        except Exception as e:
            print(f"Error during jscatter creation: {e}")
            ui.notification_show(f"Error creating plot: {e}", duration=10, type="error")
            return None
    
    @render.plot() 
    @reactive.event(input.analyze_selection)    
    def selection_plots():
        soma = soma_reactive.get()
        obs_df = obs_df_reactive.get()
        base_view = base_jscatter_view.get()

        g1_col, g1_cat = input.group1_col(), input.group1_cat()
        g2_col, g2_cat = input.group2_col(), input.group2_cat()
        analysis_col = input.analysis_col()

        if not all([soma, obs_df is not None, g1_col, g1_cat, g2_col, g2_cat, analysis_col]):
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Please define both groups and select an analysis column.", ha='center', va='center')
            ax.axis('off')
            return fig

        try:
            indices1 = obs_df.index[obs_df[g1_col].isin(g1_cat)]
            indices2 = obs_df.index[obs_df[g2_col].isin(g2_cat)]
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
            ui.notification_show(f"Error defining groups: {e}", type="error", duration=10)
            return None

        if base_view:
            selected_cells = indices1.union(indices2).tolist()
            base_view.selection(selected_cells)

        # Create slices for each group
        soma_g1 = soma.subset(obs_indices=indices1)
        soma_g2 = soma.subset(obs_indices=indices2)
        
        # For value counts, we use the pandas obs DataFrame we already have
        obs_g1_df = obs_df.loc[indices1]
        obs_g2_df = obs_df.loc[indices2]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
        fig.suptitle("Paired Group Analysis", fontsize=18)
        n_top_genes_plot = 5

        create_value_counts_barplot(obs_g1_df, analysis_col, axes[0, 0], "Group 1")
        plot_top_genes(soma_g1, n_top_genes_plot, axes[0, 1], "Group 1")
        create_value_counts_barplot(obs_g2_df, analysis_col, axes[1, 0], "Group 2")
        plot_top_genes(soma_g2, n_top_genes_plot, axes[1, 1], "Group 2")        

        return fig

# Create the Shiny app instance
app = App(app_ui, server)