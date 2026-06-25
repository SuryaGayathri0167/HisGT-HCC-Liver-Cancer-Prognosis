# Code

This folder contains all Python implementation scripts 
for the HisGT-HCC pipeline.

## Pipeline Scripts

| Script | Description |
|---|---|
| inspect_string_datasets1.py | Data inspection and STRING dataset analysis |
| check_sample_ids2.py | Sample ID verification |
| extract_histone_genes_from_epifactors3.py | Histone gene extraction from EpiFactors |
| convert_ensembl_to_gene4.py | Ensembl ID to gene symbol conversion |
| filter_histone_tcga5.py | TCGA expression filtering |
| build_histone_graph6.py | PPI graph construction from STRING |
| align_features_with_graph7.py | Feature matrix alignment with graph |
| build_graph_matrices8_optional.py | Graph matrix construction |
| top_hub_genes13.py | Hub gene identification |
| cox_analysis_correct14.py | Cox analysis and screening |
| lasso_cox_survival_model16.py | LASSO-Cox prognostic signature |
| kaplan_meier_analysis17.py | Kaplan-Meier survival analysis |
| evaluate_cindex18.py | C-index evaluation |
| train_graph_transformer20.py | Graph Transformer training |
| km_graph_transformer_validation21.py | Graph Transformer KM validation |
