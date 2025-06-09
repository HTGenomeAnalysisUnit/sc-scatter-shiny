import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import tiledbsoma.io
    import scanpy as sc
    import jscatter
    import pandas as pd
    import numpy as np
    import seaborn as sns
    return jscatter, mo, sns, tiledbsoma


@app.cell
def _(mo, tiledbsoma):
    input_h5ad_file = mo.cli_args().get("h5ad") or "/project/immune_variation/processed_data/merged_anndata/merged_pilot_data_harmony.h5ad"
    soma_path = mo.cli_args().get("soma_out") or "/project/immune_variation/processed_data/merged_anndata/merged_pilot_data_harmony_soma"
    td = tiledbsoma.io.from_h5ad(
        experiment_uri=soma_path, 
        input_path=input_h5ad_file, 
        measurement_name="RNA",
        obs_id_name="obs_id",
        var_id_name="var_id")
    return (td,)


@app.cell
def _(td, tiledbsoma):
    sc_data = tiledbsoma.Experiment.open(td)
    return (sc_data,)


@app.cell
def _(sc_data):
    table = sc_data.ms["RNA"].obsm['X_umap'].read().tables().concat().to_pandas()
    umap_df = table.pivot(
        index="soma_dim_0",  # Cell/observation index
        columns="soma_dim_1",  # UMAP dimension (0=UMAP_1, 1=UMAP_2)
        values="soma_data"  # Coordinate values
    )
    umap_df.head()
    return (umap_df,)


@app.cell
def _(sns, umap_df):
    sns.scatterplot(data=umap_df, x=0, y=1, s=1)
    return


@app.cell
def _():
    return


@app.cell
def _(df, jscatter):
    scatter = jscatter.Scatter(data=df, x='mass', y='speed')
    scatter.color(by='mass', map='plasma', order='reverse')
    scatter.opacity(by='density')
    scatter.size(by='pval', map=[2, 4, 6, 8, 10])
    scatter.background('#1E1E20')
    scatter.show()
    return


if __name__ == "__main__":
    app.run()
