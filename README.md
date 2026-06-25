# HisGT-HCC: Histone-Modifier Graph Transformer for HCC Prognosis

## Project Overview

A computational framework for hepatocellular carcinoma (HCC) 
overall survival prediction using histone-modifier gene expression 
and protein-protein interaction networks.

The pipeline integrates EpiFactors histone-modifier genes with 
STRING PPI network structure and TCGA-LIHC RNA-seq expression 
data — training a Graph Attention Network with Cox partial 
likelihood loss for patient risk stratification.

---

## Repository Structure

HisGT-HCC-Liver-Cancer-Prognosis/
├── README.md              
├── code/                  
└── presentation/          

---

## Pipeline Overview

| Stage | Description | Script |
|---|---|---|
| Stage 1 | Data inspection and STRING dataset analysis | inspect_string_datasets1.py |
| Stage 2 | Sample ID verification | check_sample_ids2.py |
| Stage 3 | Histone gene extraction from EpiFactors | extract_histone_genes_from_epifactors3.py |
| Stage 4 | Ensembl ID to gene symbol conversion | convert_ensembl_to_gene4.py |
| Stage 5 | TCGA expression filtering | filter_histone_tcga5.py |
| Stage 6 | PPI graph construction from STRING | build_histone_graph6.py |
| Stage 7 | Feature matrix alignment with graph | align_features_with_graph7.py |
| Stage 8 | Graph matrix construction | build_graph_matrices8_optional.py |
| Stage 9 | Hub gene identification | top_hub_genes13.py |
| Stage 10 | Cox analysis and screening | cox_analysis_correct14.py |
| Stage 11 | LASSO-Cox prognostic signature | lasso_cox_survival_model16.py |
| Stage 12 | Kaplan-Meier survival analysis | kaplan_meier_analysis17.py |
| Stage 13 | C-index evaluation | evaluate_cindex18.py |
| Stage 14 | Graph Transformer training | train_graph_transformer20.py |
| Stage 15 | Graph Transformer KM validation | km_graph_transformer_validation21.py |

---

## Key Results

| Model | C-index | Log-rank p |
|---|---|---|
| LASSO-Cox | 0.7478 | 4.34×10⁻¹⁶ |
| Graph Transformer | 0.7425 | 1.69×10⁻¹⁷ |

---

## Datasets Used

| Dataset | Source | Purpose |
|---|---|---|
| TCGA-LIHC RNA-seq | GDC Portal | Gene expression input |
| TCGA-LIHC Survival | UCSC Xena | Survival labels |
| TCGA-LIHC Clinical | GDC Portal | Clinical validation |
| STRING PPI v12 | string-db.org | Biological graph structure |
| EpiFactors v2.0 | epifactors.anticancer.ru | Histone-modifier gene list |

---

## Requirements

python >= 3.8
pandas
numpy
scikit-survival
torch
torch-geometric
lifelines
networkx
scipy
matplotlib
seaborn
scikit-learn

---

## Data Availability

TCGA-LIHC expression and clinical data are publicly available 
through the GDC Portal (https://portal.gdc.cancer.gov).
STRING PPI data is available at https://string-db.org.
EpiFactors database is available at https://epifactors.anticancer.ru.
Raw data files are not included in this repository due to size 
and redistribution restrictions.
