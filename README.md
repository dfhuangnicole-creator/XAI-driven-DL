<h1 align="center"> Perturbation-based DL XAI for Stylistic Analysis</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Environment-uv-6f42c1.svg" alt="uv">
  <img src="https://img.shields.io/badge/Status-Research-yellow.svg" alt="Status">
</p>

<p align="center">

## Introduction

This repository presents a research-oriented workflow for stylistic analysis of translated and non-translated English texts. The central task is binary classification, but the project is designed not only to predict labels, but also to analyze which latent or lexical signals drive those predictions.

The released implementation contains three main components:

1. a preprocessing stage that normalizes the texts and produces document-level representations
2. a VAE-based representation learning stage followed by downstream classifiers and stacking ensembles
3. a perturbation-based XAI stage for interpreting model behavior

In addition to the main VAE pipeline, the repository includes an SBERT stacking baseline and a TF-IDF baseline for comparison. Processed datasets and representative output artifacts are already versioned in the repository, so the workflow can be inspected or rerun without rebuilding every artifact from scratch.

## Visualization Results

Representative visual outputs already stored in the repository are shown below.

|                                        Confusion Matrix                                         |                                   Latent-Space Distribution                                   |
| :---------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| <img src="outputs/vae/confusion_matrix.png" alt="Classification confusion matrix" width="420"/> | <img src="outputs/vae/lda_distribution.png" alt="Latent-space LDA distribution" width="420"/> |

|                               VAE Training Curve                                |                                                                            |
| :-----------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------: |
| <img src="outputs/vae/vae_loss.png" alt="VAE training loss curve" width="420"/> |

## Experimental Results

The following table summarizes representative results from the currently committed experiment artifacts.

| Method                                 |  Accuracy  |  Macro F1  | Artifact                                                |
| :------------------------------------- | :--------: | :--------: | :------------------------------------------------------ |
| VAE stacking on latent features        | **0.8778** | **0.8778** | `outputs/vae/stacking_report.txt`                       |
| Logistic regression on latent features |   0.7200   |   0.7200   | `outputs/vae/log_classification_report.txt`             |
| SBERT stacking on sentence embeddings  | **0.9778** | **0.9777** | `outputs/vae/sbert_stacking_report_20260320_154645.txt` |

The repository also includes interpretation-oriented outputs such as:

- `outputs/vae/perturbation_analysis.xlsx`
- `outputs/vae/perturbation_results.json`
- `outputs/vae/interpretation_report.json`

## Reproducibility and Experimental Setup

The settings below describe the default configuration implemented in the released codebase and are intended to make reruns easier to reproduce.

### 1. Data and Preprocessing

| Item                     | Setting                                                                |
| :----------------------- | :--------------------------------------------------------------------- |
| Task                     | Binary classification of translated English vs. non-translated English |
| Input source             | Excel file placed at `data/raw/data.xlsx`                              |
| Train/test split         | 80% / 20%                                                              |
| Random seed              | 42                                                                     |
| Text normalization       | contraction expansion, lowercasing, whitespace normalization           |
| Lexical control          | label-conditional neutralization with class-specific word lists        |
| Entity masking           | `PERSON`, `ORG`, `GPE`, `LOC`, `NORP`, and `FAC` entities are masked   |
| Sentence encoder         | `sentence-transformers/all-mpnet-base-v2`                              |
| Segment length           | maximum 350 tokenizer tokens per segment                               |
| Document representation  | mean pooling across segment embeddings                                 |
| Feature scaling          | StandardScaler is applied before downstream classifiers                |
| TF-IDF baseline features | `max_features=4096`, `ngram_range=(1, 3)`                              |

### 2. VAE Training Configuration

| Item                | Setting                                                        |
| :------------------ | :------------------------------------------------------------- |
| Model family        | symmetric MLP VAE                                              |
| Input dimension     | 768                                                            |
| Encoder widths      | 256 -> 128 -> 50                                               |
| Latent dimension    | 50                                                             |
| Latent heads        | separate mean and log-variance projections                     |
| Decoder             | mirrored batch-normalized MLP decoder                          |
| Normalization       | BatchNorm1d                                                    |
| Activation          | LeakyReLU with negative slope 0.2                              |
| Dropout             | 0.1 after the first encoder layer                              |
| Reconstruction loss | mean squared error                                             |
| KL term             | variational KL divergence with annealing and free-bits control |
| Optimizer           | AdamW                                                          |
| Learning rate       | 1e-4                                                           |
| Batch size          | 64                                                             |
| Epochs              | 100                                                            |
| KL weighting        | `beta=0.5`                                                     |
| Free bits           | `0.1`                                                          |
| KL annealing        | enabled, annealed over 20 epochs                               |
| Training monitoring | total loss, test loss, and KL divergence curves                |
| Hardware            | single NVIDIA GPU                                              |

### 3. Downstream Classifier Configuration

| Pipeline              | Configuration                                                                                                                                                                                                                                      |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VAE stacking          | SVM with `C=20`, `kernel=rbf`, `gamma=scale`; Decision Tree with `max_depth=5`, `criterion=gini`, `min_samples_leaf=12`; KNN with `n_neighbors=3`, `metric=manhattan`; Logistic Regression meta-classifier with `max_iter=1000`; stacking `cv=5`   |
| SBERT stacking        | Linear SVM with `C=0.1`; Random Forest with `n_estimators=1000`, `max_depth=20`, `min_samples_split=5`, `bootstrap=False`; KNN with `n_neighbors=3`, `metric=manhattan`; Logistic Regression meta-classifier with `max_iter=1000`; stacking `cv=5` |
| VAE logistic baseline | Logistic Regression with `max_iter=100`, `C=1`, `random_state=42`                                                                                                                                                                                  |
| TF-IDF model search   | Stacking combinations evaluated over SVM, Random Forest, XGBoost, KNN, and Decision Tree with `cv=5`                                                                                                                                               |

### 4. Reproducibility Checklist

| Item                                            | Status                 |
| :---------------------------------------------- | :--------------------- |
| Processed datasets included in repository       | Yes                    |
| Saved experiment reports included in repository | Yes                    |
| Saved VAE weights included in repository        | Yes                    |
| Default hyperparameters documented              | Yes                    |
| Random seed documented                          | Yes                    |
| Hardware class documented                       | Yes, single NVIDIA GPU |

### 5. Computational Profile

The released artifacts indicate a relatively modest experimental scale.

| Item                        | Value                                                  |
| :-------------------------- | :----------------------------------------------------- |
| Total processed samples     | 900                                                    |
| Training samples            | 720                                                    |
| Test samples                | 180                                                    |
| VAE parameter count         | 482,524                                                |
| Primary embedding backbone  | all-mpnet-base-v2                                      |
| Dominant preprocessing cost | SBERT inference over text segments                     |
| Dominant training cost      | VAE optimization and 5-fold stacked classifier fitting |

In practice, preprocessing is the most expensive stage because each document is segmented and encoded with Sentence-BERT before mean pooling. The VAE itself is comparatively small, and the downstream classifiers operate on standardized embedding features or latent features over a 720-sample training split. Accordingly, the overall computational burden is moderate for a single-GPU research workflow rather than large-scale deep training.

Exact wall-clock training time is not yet versioned as a stable benchmark in this repository because it depends on local hardware availability, storage throughput, and whether external model assets are already cached.

## Usage Instructions

### 1. Environment Installation

The project uses `uv` for environment and dependency management.

```bash
uv sync
uv run xai-style check
```

If the local spaCy model is missing, install it explicitly:

```bash
uv run python -m spacy download en_core_web_sm
```

### 2. Data Preparation

If you have the original Excel file, place it at:

```text
data/raw/data.xlsx
```

The repository already contains the processed files used by the current pipeline:

- `data/processed/vae/processed_final.json`
- `data/processed/tfidf/processed_TFIDF_final.json`
- `data/processed/tfidf/feature_names.json`

To regenerate the processed artifacts locally:

```bash
uv run xai-style fetch-sbert-model
uv run xai-style vae-preprocess
uv run xai-style tfidf-preprocess
```

### 3. Training and Evaluation

Run the main VAE-based workflow:

```bash
uv run xai-style fetch-sbert-model
uv run xai-style vae-preprocess
uv run xai-style vae-train
uv run xai-style vae-search
uv run xai-style vae-vote
uv run xai-style vae-final
uv run xai-style vae-plot
uv run xai-style vae-perturb
```

Run the baseline workflows:

```bash
uv run xai-style vae-logistic
uv run xai-style vae-sbert-search
uv run xai-style vae-sbert-vote
uv run xai-style vae-sbert-final
uv run xai-style tfidf-preprocess
uv run xai-style tfidf-vote
```

### 4. Verification

Run the test suite:

```bash
uv run pytest
```

## Project Structure

```text
XAI-driven-DL/
├── data/                         # Raw and processed data
├── outputs/                      # Figures, reports, and saved models
├── tests/                        # CLI, dataset, and smoke tests
├── tools/                        # Standalone preprocessing and baseline scripts
├── xai_for_stylistic_analysis/   # Main package and VAE pipeline
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # Locked dependency resolution
└── README.md                     # Project documentation
```

## Acknowledgements

This project is motivated by research on translationese, representation learning for text classification, and explainable AI for stylistic analysis.

The implementation builds on open-source libraries including PyTorch, scikit-learn, sentence-transformers, spaCy, Hugging Face, and XGBoost.
