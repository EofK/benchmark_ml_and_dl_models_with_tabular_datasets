# Benchmarking 12 Deep Learning Models for Tabular Binary Classification

This repository contains the code, results, and research report for a large-scale benchmarking study evaluating 12 deep learning models across 182 tabular binary classification datasets.

## Study Overview

This study systematically evaluates 12 deep learning architectures from the `pytorch_tabular` and `pytorch_widedeep` libraries, spanning four model families:

- **Transformer-based:** FT-Transformer, TabTransformer, SAINT, Self-Attention MLP, TabFastFormer
- **Tree-Neural Hybrid:** GANDALF, GATE, NODE
- **Feature Interaction/Attention:** AutoInt, DANet
- **MLP-based:** Category Embedding, TabMLP

Models were evaluated on 182 publicly available tabular binary classification datasets from OpenML, UCI, Kaggle, and other sources. All models used fixed default hyperparameters and consistent training configurations to ensure fair comparison.

## Key Findings

- 11 of 12 models form a statistically equivalent top tier on accuracy, with mean accuracy ranging from 77.1% to 83.8%
- Category Embedding and TabMLP are the strongest all-around choices when both accuracy and throughput are considered
- FT-Transformer leads on dataset-level win rate but is notably slower, presenting a tradeoff for latency-sensitive applications
- Neighborhood complexity measures (N1, N3, N4) are the strongest predictors of classification difficulty
- Traditional ML models still outperform the best DL models on tabular data, but the gap narrows at mid-tier ML

## Repository Contents
├── notebooks/ 

│ └── dl_benchmark.ipynb # Main benchmarking notebook (Google Colab) 

├── results/

│ └── ml_dl_results_consolidation.xlsx # Full model-dataset results 

├── report/

│ └── DL_Binary_Classification_Final_Paper.docx # Final research report 

│   └── complexity_measurement.ipynb    # Dataset complexity measurement (run locally, not on Colab)

└── README.md

## How to Use This Notebook

1. **Setup:** Upload the notebook to Google Colab.
2. **Configure Paths:** In the first code cell, update the `USER_PATH` variable to point to your project folder:
```python
   USER_PATH = '/content/drive/MyDrive/your_folder_name'
```
3. **Mount Google Drive:** Run the drive mount cell to connect your Google Drive.
4. **Run Cells Sequentially:** Execute each cell in order. GPU runtime is required and strongly recommended (A100 preferred).

## Complexity Measurement Notebook

The `complexity_measurement.ipynb` notebook calculates the 22 Lorena et al. complexity measures for each dataset using the `problexity` Python library. This notebook is designed to run locally on a CPU-equipped machine, not on Google Colab.

> **Warning:** The `clsCoef` measure can take several hours to compute on datasets with high proportions of binary or categorical features. On one dataset in this study it required over five hours to complete. Monitor runtimes closely and consider running clsCoef separately from the other 21 measures if compute time is a constraint.

Install the required library:
```bash
pip install problexity
```

## Requirements

- Python 3.13+
- Google Colab Pro+ (A100 GPU recommended)
- pytorch_tabular
- pytorch_widedeep
- scikit-learn
- pandas
- numpy
- optuna

Install dependencies:
```bash
pip install pytorch_tabular pytorch_widedeep scikit-learn pandas numpy
```

## Datasets

All 182 datasets are publicly available from:
- [OpenML](https://www.openml.org) (131 datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu) (30 datasets)
- [Kaggle](https://www.kaggle.com) (15 datasets)
- Other sources (6 datasets)

A full dataset catalog including sources, record counts, feature counts, and complexity scores is provided in Appendix B of the research report.

## Citation

If you use this code or findings in your work, please cite:

Kaempf, E. R. (2026). Benchmarking 12 Deep Learning Models for Binary Classification: Accuracy, Complexity, and Speed.

## Related Work

This study extends a companion ML benchmark:

Kaempf, E. R. (2025). Benchmarking 15 Machine Learning Models for Binary
Classification: Accuracy, Complexity, and Speed.

## License

This project is released for research and educational use. Dataset licenses vary by source — please refer to each dataset's original repository for usage terms.
