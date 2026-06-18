# Chinese News Classification and Clustering

## Project Overview

This historical coursework project compares classical text representations and machine-learning methods for Chinese news classification and clustering. The original course submission used a THUCNews/cnews subset reported to contain approximately 50,000 Chinese news articles across 10 categories.

This repository preserves the original course scripts and organizes the reported methods, figures, and selected results for portfolio presentation.

> This repository is a cleaned presentation copy of a historical coursework project. The reported results are taken from the original course submission and are currently undergoing reproducibility verification. The scripts are archived for transparency, but the current Phase 1A release does not claim end-to-end reproducibility.

## Dataset Note

The dataset is not included in this repository. Raw news articles are not redistributed. Anyone wishing to examine or run the archived scripts in a later reproducibility phase must obtain an appropriate THUCNews/cnews dataset separately and review its applicable terms.

See [`data/README.md`](data/README.md) for the current data policy and the planned local data location.

## Methods Reported in the Original Course Submission

### Text representations

- TF-IDF
- Word2Vec
- Doc2Vec

### Classification models

- Linear Support Vector Machine (Linear SVM)
- Naive Bayes
- Logistic Regression

### Clustering models

- KMeans
- DBSCAN

### Evaluation metrics

- Classification: Accuracy and Macro-F1
- Clustering: Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI)

## Key Reported Results

The following values are transcribed from the original course report. They have not been regenerated during Phase 1A.

| Task | Representation | Model | Metric | Reported value |
|---|---|---|---|---:|
| Classification | TF-IDF | Linear SVM | Accuracy | 82.39% |
| Classification | TF-IDF | Linear SVM | Macro-F1 | 82.65% |
| Classification | TF-IDF | Logistic Regression | Accuracy | 82.13% |
| Classification | TF-IDF | Logistic Regression | Macro-F1 | 82.47% |
| Clustering | TF-IDF | KMeans | NMI | 0.1740 |

Source: original course report. A machine-readable copy is available in [`results/metrics.csv`](results/metrics.csv).

## Existing Figures

The files under `results/figures/` are archived figures from the original coursework project. Phase 1A does not regenerate them.

- `classification_comparison.png`
- `cluster_ARI.png`
- `cluster_NMI.png`
- `text_length_distribution.png`

## Repository Structure

```text
chinese-news-classification-clustering/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── data_stats.py
│   ├── classify.py
│   ├── cluster.py
│   └── utils.py
├── results/
│   ├── figures/
│   └── metrics.csv
└── report/
    └── README.md
```

The four files under `src/` are byte-for-byte copies of the original coursework scripts and have not been edited during Phase 1A.

## Reproducibility Status

Phase 1A is documentation and presentation cleanup only. During this phase:

- no Python scripts were executed;
- no models were trained;
- no experiments were rerun;
- no figures or metrics were regenerated;
- no algorithm, path, or dependency issue was corrected; and
- the historical dependency versions were documented but not re-validated.

The archived scripts may require data-path, dependency, and implementation review before they can be run reliably. That work is outside the Phase 1A scope.

## Limitations

- The project reports results from one Chinese news dataset and does not establish general performance across domains.
- The original comparison focuses on classical machine-learning and shallow embedding methods rather than Transformer-based models.
- Raw data are not included, so this presentation release cannot reproduce results on its own.
- The Word2Vec, Doc2Vec, clustering, and path implementations remain exactly as they appeared in the historical project and have not been corrected or re-validated.
- The reported values should be understood as historical course-submission results, not as a newly verified benchmark.

## Course Report

The original report contains personal and course-administration information and is not included. See [`report/README.md`](report/README.md) for the publication policy.
