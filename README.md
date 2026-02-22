# Benchmarking Machine Learning and Deep Learning Models with Tabular Datasets

A comprehensive benchmarking study evaluating classical machine learning and deep learning models on tabular binary classification tasks. This project provides reproducible code, systematic methodology, and detailed analysis of model performance across 159 diverse datasets.

## 📊 Project Overview

This repository contains the complete workflow for benchmarking 15+ machine learning model families on binary classification tasks. The study addresses four core research questions:

1. **Which models perform best overall?**
2. **What makes datasets difficult to classify?**
3. **Which models handle specific complexity types most effectively?**
4. **How do accuracy and speed trade off across models?**

### Key Features

- **Comprehensive Evaluation**: 2,384+ model-dataset combinations
- **Complexity Analysis**: 22 dataset complexity measures based on Lorena et al. (2019) framework
- **Performance Metrics**: Accuracy, F1-score, precision, recall, and throughput measurements
- **Reproducible Pipeline**: End-to-end Jupyter notebook workflow
- **Deep Learning Support**: PyTorch neural network implementations for tabular data
- **Hyperparameter Optimization**: Systematic tuning using GridSearchCV

## 🗂️ Repository Structure

```
binary_eval/
├── README.md                           ← You are here
├── config.py                           ← Centralized configuration and paths
├── requirements.txt                    ← Python dependencies
│
├── datasets/                           ← Data storage
│   ├── raw/                           ← Original datasets (CSV format)
│   ├── processed/                     ← Preprocessed datasets (Parquet)
│   │   ├── benchmark_parquets/       ← Optimized format for model training
│   │   └── metadata/                  ← Schema and transformation metadata
│
├── notebooks/                          ← Analysis pipeline (numbered sequence)
│   ├── 01_load_and_explore.ipynb      ← Data loading and EDA
│   ├── 02_diagnose_features.ipynb     ← Feature quality assessment
│   ├── 03_scale_and_encode_features.ipynb  ← Data transformation
│   ├── 04_scale_encode_export.ipynb   ← Export preprocessed data
│   ├── 05_prepare_model_baselines.ipynb    ← Baseline model performance
│   ├── 06_tune_hyperparameters.ipynb  ← Hyperparameter optimization
│   ├── 07_run_model_evaluation.ipynb  ← Model training and evaluation
│   ├── 08_calculate_dataset_complexity.ipynb  ← Complexity metrics
│   └── 19_display_results.ipynb       ← Results visualization
│
├── outputs/                            ← Generated outputs
│   ├── baseline_results/              ← Pre-tuning performance
│   ├── tuned_models/                  ← Optimized model configurations
│   ├── tuning_logs/                   ← Hyperparameter search logs
│   ├── results/                       ← Final benchmark results
│   ├── figures/                       ← Visualizations and plots
│   ├── summary/                       ← Summary statistics
│   └── run_manifests/                 ← Execution metadata (JSON)
│
└── project_clean/                      ← Production-ready notebooks
    ├── 04_scale_encode_export.ipynb
    ├── 05_prepare_model_baselines.ipynb
    ├── 06_tune_hyperparameters.ipynb
    ├── 07_run_model_evaluation.ipynb
    └── 08_calculate_dataset_complexity.ipynb
```

## 🤖 Models Evaluated

### Classical Machine Learning
- **Linear Models**: Logistic Regression, Ridge Classifier
- **Tree-Based**: Decision Tree, Random Forest, Extra Trees
- **Boosting**: AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Support Vector Machines**: Linear SVC, SVC (RBF kernel)
- **Nearest Neighbors**: K-Nearest Neighbors
- **Naive Bayes**: Gaussian Naive Bayes
- **Discriminant Analysis**: Linear Discriminant Analysis, Quadratic Discriminant Analysis

### Deep Learning
- **Neural Networks**: PyTorch tabular models with GPU acceleration
- **Embedding Layers**: Support for categorical features
- **Custom Architectures**: Configurable depth and width

## 📈 Dataset Complexity Measures

Using the `problexity` library (Lorena et al., 2019 framework):

- **Feature-Based**: F1 (Maximum Fisher's Discriminant Ratio), F2, F3, F4
- **Linearity**: L1, L2, L3
- **Neighborhood**: N1, N2, N3, N4
- **Network**: Density, ClsCoef, Hubs
- **Dimensionality**: T2, T3, T4
- **Class Imbalance**: C1, C2
- **Feature Correlation and Overlap measures**

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for deep learning acceleration)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/EofK/benchmark_ml_and_dl_models_with_tabular_datasets.git
cd benchmark_ml_and_dl_models_with_tabular_datasets
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure paths**
Edit `config.py` to set your local paths:
```python
PROJECT_BASE = Path("C:/Misc/binary_eval")  # Update this path
```

### Usage

1. **Data Preparation**: Run notebooks 01-04 to load, explore, and preprocess datasets
2. **Model Baseline**: Execute notebook 05 to establish baseline performance
3. **Hyperparameter Tuning**: Run notebook 06 for optimization
4. **Model Evaluation**: Execute notebook 07 for comprehensive benchmarking
5. **Complexity Analysis**: Run notebook 08 to calculate dataset complexity
6. **Results Visualization**: Use notebook 19 to generate plots and summaries

## 📊 Results

Key findings from the benchmarking study:

- **Best Overall Models**: XGBoost, LightGBM, and Random Forest consistently achieve top performance
- **Speed-Accuracy Tradeoffs**: Logistic Regression and Linear SVC offer excellent throughput with competitive accuracy
- **Complexity Insights**: Dataset difficulty correlates strongly with neighborhood cohesion (N1) and class overlap (F1)
- **Deep Learning**: Neural networks excel on larger datasets (>50K samples) with proper regularization

Detailed results are available in the `outputs/summary/` directory.

## 🔬 Methodology

### Data Preprocessing
- **Scaling**: StandardScaler for numerical features
- **Encoding**: One-hot encoding for categorical features
- **Missing Values**: Median imputation for numerical, mode for categorical
- **Format**: Parquet files for efficient storage and loading

### Model Training
- **Cross-Validation**: Stratified 5-fold CV for hyperparameter tuning
- **Train/Test Split**: 80/20 stratified split
- **Random Seeds**: Fixed seeds for reproducibility
- **Metrics**: Accuracy, weighted F1, precision, recall, ROC-AUC

### Performance Measurement
- **Accuracy Metrics**: Multiple classification metrics
- **Throughput**: Predictions per second on test set
- **Memory Usage**: Peak memory consumption during training

## 🔧 Technologies Used

- **Core**: Python 3.13, Pandas, NumPy
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch 2.x with CUDA support
- **Complexity**: problexity library
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Storage**: Parquet (via PyArrow)
- **Notebooks**: Jupyter, VS Code

## 📝 Reproducibility

All experiments use:
- Fixed random seeds (42)
- Versioned dependencies (`requirements.txt`)
- Documented preprocessing steps
- Saved model configurations
- Execution metadata in JSON manifests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Ed Kaempf**
- Email: edkaempf@gmail.com
- GitHub: [@EofK](https://github.com/EofK)
- LinkedIn: [Ed Kaempf](https://www.linkedin.com/in/ed-kaempf-4887839b/)

## 🙏 Acknowledgments

- Lorena et al. (2019) for the dataset complexity framework
- scikit-learn community for excellent ML tools
- PyTorch team for deep learning infrastructure
- UCI Machine Learning Repository and OpenML for datasets

## 📚 Citation

If you use this repository in your research, please cite:

```
Ed Kaempf (2026). Benchmarking Machine Learning and Deep Learning Models 
for Binary Classification: Accuracy, Complexity, and Speed. 
GitHub repository: https://github.com/EofK/benchmark_ml_and_dl_models_with_tabular_datasets
```

## 🔄 Project Status

Active development - Last updated: February 2026

## 📞 Contact

For questions, suggestions, or collaboration opportunities:
- Open an issue on GitHub
- Email: edkaempf@gmail.com

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
