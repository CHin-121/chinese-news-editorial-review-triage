# Chinese News Auto-Categorization and Editorial Review Triage

## Project Overview

This project reframes a historical Chinese news classification and clustering coursework project as a data science case study for **automatic news categorization** and **editorial review prioritization**.

The practical problem is:

> Given a large collection of Chinese news articles, how can a news platform automatically assign articles to categories while also identifying low-confidence predictions that should be prioritized for human editorial review?

The original coursework compared classical text representations and machine-learning methods for Chinese news classification and clustering. I later extended it into a reproducibility-audited text analytics project with error diagnosis and decision-margin-based review triage.

## Key Findings

* The dataset contains **50,000 Chinese news articles** across **10 balanced categories**.
* The test set contains **10,000 articles**.
* The best reproduced classification model was **TF-IDF + Linear SVM**.
* TF-IDF + Linear SVM achieved **82.39% Accuracy** and **82.65% Macro-F1**.
* Four historical classification metrics from the original course report were reproduced at reported precision.
* Error analysis identified high-risk confusion pairs, including **家居 → 时尚** and **时尚 → 家居**.
* Reviewing the lowest-confidence **10%** of articles captured **35.72%** of all classification errors.
* Reviewing the lowest-confidence **20%** of articles captured **60.08%** of all classification errors.
* Direct TF-IDF + KMeans did not reproduce the historical NMI result, so clustering is treated as a diagnostic appendix rather than the main contribution.

## Project Phases

### Phase 1A: Presentation Cleanup

Organized the historical coursework scripts, selected reported results, archived figures, dataset policy, and reproducibility disclaimer for portfolio presentation.

### Phase 1B: Reproducibility Audit

Added an independent reproducibility pipeline to validate the local dataset, regenerate selected TF-IDF classification results, and compare regenerated metrics with the original course report.

### Phase 1C: Editorial Review Triage Analysis

Extended the classification model into a decision-support workflow. Linear SVM decision margins were used to rank low-confidence predictions for potential human editorial review.

## Practical Data Science Framing

The project is not only a text classification exercise. It follows a complete data science workflow:

1. validate a Chinese text dataset;
2. transform Chinese news text into TF-IDF features;
3. train supervised classification models;
4. reproduce and audit historical metrics;
5. diagnose errors using class-level reports and confusion matrices;
6. rank low-confidence predictions for human review;
7. document limitations and avoid overclaiming reproducibility.

## Dataset Note

The dataset is **not included** in this repository. Raw news articles are not redistributed.

The original coursework used a THUCNews/cnews-style subset reported to contain approximately 50,000 Chinese news articles across 10 categories. Anyone wishing to run the scripts must obtain an appropriate dataset separately and review its applicable terms.

See [`data/README.md`](data/README.md) for the current data policy and the planned local data location.

## My Role

I completed the original coursework project independently, including data exploration, text representation, model comparison, clustering analysis, result visualization, and report writing.

In the later portfolio phase, I further organized the repository, added a reproducibility audit pipeline, regenerated the historical TF-IDF classification results, and extended the project with editorial review triage analysis based on Linear SVM decision margins.

## Portfolio Relevance

This project demonstrates my ability to turn a Chinese text classification coursework project into a practical data science case study.

It covers:

* Chinese text preprocessing and feature engineering;
* supervised text classification;
* reproducibility auditing;
* classification metric interpretation;
* error diagnosis through confusion analysis;
* human-in-the-loop review prioritization;
* clear documentation of data and reproducibility limitations.

For data science master’s applications, this project highlights my ability to connect Chinese-language text analysis with measurable model evaluation and decision-oriented analytics.

## Methods

### Text Representation

* **TF-IDF** was used as the main text representation.
* The historical coursework also reported Word2Vec and Doc2Vec experiments, but Phase 1B and Phase 1C focus on TF-IDF because it was the strongest and most stable representation in the reproduced classification results.

### Classification Models

* **Linear SVM** was used as the main classification model.
* **Logistic Regression** was used as a comparison model.
* Historical coursework also reported Naive Bayes, but it is not part of the Phase 1B/1C main workflow.

### Clustering Models

* Historical coursework reported KMeans and DBSCAN.
* In the reproducibility audit, direct TF-IDF + KMeans did not reproduce the historical NMI value.
* Therefore, clustering is treated as a diagnostic appendix, not as the central contribution of this repository.

### Editorial Review Triage

The Phase 1C analysis uses the Linear SVM decision scores.

For each article, the model produces a score for each candidate category. The review margin is defined as:

```text
review_margin = top1_decision_score - top2_decision_score
```

A smaller margin means the model is less confident because the top two category scores are close. These low-margin articles are prioritized for human editorial review.

This margin is **not a probability**. It is used only as a ranking signal for review prioritization.

## Main Results

### Classification Results

| Model               | Representation | Accuracy | Macro-F1 | Status                           |
| ------------------- | -------------- | -------: | -------: | -------------------------------- |
| Linear SVM          | TF-IDF         |   82.39% |   82.65% | Reproduced at reported precision |
| Logistic Regression | TF-IDF         |   82.13% |   82.47% | Reproduced at reported precision |

### Editorial Review Triage Results

The test set contains 10,000 articles. The Linear SVM model made 1,761 classification errors.

| Review budget | Reviewed articles | Captured errors | Share of all errors captured | Errors found per 100 reviewed |
| ------------: | ----------------: | --------------: | ---------------------------: | ----------------------------: |
|            5% |               500 |             326 |                       18.51% |                          65.2 |
|           10% |             1,000 |             629 |                       35.72% |                          62.9 |
|           20% |             2,000 |           1,058 |                       60.08% |                          52.9 |

Key interpretation:

> Reviewing only the lowest-confidence 10% of articles captured 35.72% of all classification errors.

This shows that the decision-margin signal can help prioritize human review resources toward higher-risk predictions.

### Clustering Diagnostic

| Method                 | Historical NMI | Regenerated NMI | Status         |
| ---------------------- | -------------: | --------------: | -------------- |
| Direct TF-IDF + KMeans |         0.1740 |          0.1883 | Not reproduced |

The mismatch suggests that the historical KMeans configuration may have involved unrecorded details, such as feature transformation or implementation differences. The clustering result is therefore documented as a diagnostic finding rather than a verified reproduced result.

## Error Analysis

The project includes class-level error diagnosis through:

* per-class precision, recall, F1, and support;
* a 10 × 10 confusion matrix;
* top confusion pairs ranked by error count;
* review triage curves under 5%, 10%, and 20% review budgets.

Example high-risk confusion pairs include:

| True label | Predicted label | Error count |
| ---------- | --------------- | ----------: |
| 家居         | 时尚              |         163 |
| 时尚         | 家居              |         117 |
| 时政         | 家居              |         108 |

These results show that model errors are not evenly distributed. Some category pairs create higher review risk and deserve closer inspection.

## Repository Structure

```text
chinese-news-classification-clustering/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── clean_baseline.json
│   └── historical_compat.json
├── data/
│   └── README.md
├── docs/
│   ├── algorithm-audit.md
│   ├── editorial-triage-analysis.md
│   └── reproducibility.md
├── report/
│   └── README.md
├── results/
│   ├── figures/
│   ├── metrics.csv
│   └── reproduced/
│       ├── 20260619T193149Z/
│       │   ├── comparison.csv
│       │   ├── metrics.csv
│       │   └── run_metadata.json
│       └── 20260620T101144Z/
│           ├── confusion_matrix.csv
│           ├── per_class_report.csv
│           ├── review_priority_summary.json
│           ├── review_triage_curve.csv
│           └── top_confusion_pairs.csv
└── src/
    ├── classify.py
    ├── cluster.py
    ├── data_stats.py
    ├── utils.py
    └── repro_pipeline/
        ├── __init__.py
        ├── classification.py
        ├── cli.py
        ├── clustering.py
        ├── data.py
        └── reporting.py
```

The original four coursework scripts under `src/` are preserved as historical files. The newer `src/repro_pipeline/` modules were added for reproducibility auditing and editorial review triage analysis.

## Reproducibility Status

Phase 1A was documentation and presentation cleanup only. No scripts were executed and no results were regenerated during that phase.

Phase 1B added an independent reproducibility audit pipeline. The historical TF-IDF + Linear SVM and TF-IDF + Logistic Regression classification metrics were regenerated and matched the original course report at reported precision.

Phase 1C added editorial review triage analysis. It used Linear SVM decision margins to prioritize low-confidence predictions for human review.

Raw news data are not included in this repository, so the public repository does **not** claim standalone end-to-end reproducibility. The repository documents the pipeline, metadata, and regenerated outputs, but anyone wishing to run the scripts must obtain the appropriate dataset separately.

## Historical Coursework Materials

### Methods Reported in the Original Course Submission

The original coursework compared the following methods:

**Text representations**

* TF-IDF
* Word2Vec
* Doc2Vec

**Classification models**

* Linear Support Vector Machine
* Naive Bayes
* Logistic Regression

**Clustering models**

* KMeans
* DBSCAN

**Evaluation metrics**

* Classification: Accuracy and Macro-F1
* Clustering: Adjusted Rand Index and Normalized Mutual Information

### Historical Reported Results

The following values were transcribed from the original course report.

| Task           | Representation | Model               | Metric   | Reported value |
| -------------- | -------------- | ------------------- | -------- | -------------: |
| Classification | TF-IDF         | Linear SVM          | Accuracy |         82.39% |
| Classification | TF-IDF         | Linear SVM          | Macro-F1 |         82.65% |
| Classification | TF-IDF         | Logistic Regression | Accuracy |         82.13% |
| Classification | TF-IDF         | Logistic Regression | Macro-F1 |         82.47% |
| Clustering     | TF-IDF         | KMeans              | NMI      |         0.1740 |

A machine-readable copy of the historical reported values is available in [`results/metrics.csv`](results/metrics.csv).

## Existing Figures

The files under `results/figures/` are archived figures from the original coursework project. Phase 1A did not regenerate them.

* `classification_comparison.png`
* `cluster_ARI.png`
* `cluster_NMI.png`
* `text_length_distribution.png`

Newer Phase 1B and Phase 1C regenerated results are stored separately under `results/reproduced/`.

## Limitations

* The project uses one Chinese news dataset and does not establish general performance across all domains.
* Raw data are not included, so the repository cannot be run end-to-end by itself.
* The review-margin triage strategy is based on Linear SVM decision scores rather than calibrated probabilities.
* The editorial review analysis is a simulation using test-set labels, not a live newsroom deployment.
* The original comparison focused on classical machine-learning and shallow embedding methods rather than Transformer-based models.
* Word2Vec and Doc2Vec were reported in the original coursework but were not rerun in the Phase 1B/1C workflow.
* Direct TF-IDF + KMeans did not reproduce the historical NMI result, and the exact historical clustering configuration remains ambiguous.
* The project should be understood as a reproducibility-audited coursework extension and portfolio case study, not as a production news classification system.

## Course Report

The original course report contains personal and course-administration information and is not included.

See [`report/README.md`](report/README.md) for the publication policy.
