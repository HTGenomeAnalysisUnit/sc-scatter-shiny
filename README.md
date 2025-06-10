# singlecell interactive plot with jupyter scatter

Currently just a prototype.

Main concept is to exploit [jupyter scatter](https://jupyter-scatter.dev/), tileDB-SOMA and tileDB-VCF to allow scalable interactive visualization of large-scale single-cell data.

The app is built using pyshiny and can be published to R Studio Connect.

As a starting point, we want to be able to:

- Load h5ad or tileDB-SOMA data from a pre-defined path
- Generate linked views representing UMAP plots colored by column of interest from obs or genes of interest (limited number)
- Allow groups definition based on cell annotations
- Compare expression of genes of interest or distribution of annotations of interest across groups making barplots and violin plots
- Be able to add custom annotations for samples and/or cells
- Be able to extract genotypes for a SNP of interest and inject them as annotations to stratify groups or color code the plots.

## Limitations so far

- We are unable to capture interactive selections directly from plots
- Linked selection across views is not working properly 