# Chinese News Auto-Categorization and Editorial Review Triage

## Project Overview

This project reframes a historical Chinese news classification and clustering coursework project as a data science case study for automatic news categorization and editorial review prioritization.

The practical problem is: given a large set of Chinese news articles, how can a news platform automatically assign articles to categories while also identifying low-confidence predictions that should be prioritized for human editorial review?

## Key Findings

- The dataset contains 50,000 Chinese news articles across 10 balanced categories.
- The TF-IDF + Linear SVM model achieved 82.39% Accuracy and 82.65% Macro-F1 on a 10,000-article test set.
- Four historical classification metrics from the original course report were reproduced at reported precision.
- Error analysis identified high-risk confusion pairs, including 家居 → 时尚 and 时尚 → 家居.
- Reviewing the lowest-confidence 10% of articles captured 35.72% of all classification errors.
- Reviewing the lowest-confidence 20% of articles captured 60.08% of all classification errors.
- Direct TF-IDF + KMeans did not reproduce the historical NMI result, so clustering is treated as a diagnostic appendix.

## Project Phases

- **Phase 1A: Presentation cleanup**  
  Organized the historical coursework scripts, selected results, dataset policy, and reproducibility disclaimer.

- **Phase 1B: Reproducibility audit**  
  Added an independent reproducibility pipeline and reproduced the historical TF-IDF classification results.

- **Phase 1C: Editorial review triage analysis**  
  Used Linear SVM decision margins to prioritize low-confidence articles for human review.

## Dataset Note

The dataset is not included in this repository. Raw news articles are not redistributed. Anyone wishing to examine or run the archived scripts in a later reproducibility phase must obtain an appropriate THUCNews/cnews dataset separately and review its applicable terms.

See [`data/README.md`](data/README.md) for the current data policy and the planned local data location.

## My Role

I completed this coursework project independently, including data exploration, text representation, model comparison, clustering analysis, result visualization, and report writing.

## Portfolio Relevance

This project demonstrates my ability to organize a standard text-mining workflow, including dataset description, feature extraction, supervised classification, unsupervised clustering, model evaluation, result visualization, and limitation reporting.


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

