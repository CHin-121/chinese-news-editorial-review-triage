# Phase 1B-minimal Algorithm Audit

## Scope

This audit covers only:

- TF-IDF + Linear SVM classification;
- TF-IDF + Logistic Regression classification; and
- direct TF-IDF + KMeans clustering.

The four historical scripts remain unchanged. Word2Vec, Doc2Vec, the historical SVD clustering branch, DBSCAN, and BERT were not executed or corrected.

## Classification Audit

### Historical compatibility track

The compatibility track reproduces the behavior of the archived TF-IDF classification section:

- an 80/20 stratified split with `random_state=42`;
- TF-IDF fitted on the complete outer training split;
- `max_features=5000`, `ngram_range=(1, 2)`, and `min_df=5`;
- five-fold `GridSearchCV` optimized for Macro-F1;
- `LinearSVC(dual=False)` and `LogisticRegression(solver="liblinear", max_iter=500)`; and
- `C` searched over `0.1`, `0.5`, `1.0`, and `2.0`.

The held-out test set is not used to fit TF-IDF or tune model parameters. However, TF-IDF is fitted before the internal five-fold search, so each validation fold contributes to the vocabulary and IDF statistics used by the other folds. This is an inner-cross-validation leakage issue, not outer-test leakage.

### Clean baseline track

The clean baseline places TF-IDF and the classifier in a scikit-learn `Pipeline`. TF-IDF is therefore fitted independently inside each cross-validation fold. After model selection, GridSearchCV refits the complete pipeline on the full outer training split before evaluating the untouched test split.

The clean track selected the same `C` values and produced the same held-out test metrics as the compatibility track. This equality does not make the two procedures algorithmically identical; it means the corrected CV boundary did not change model selection for this dataset and parameter grid.

### Classification result

| Track | Model | Best C | Accuracy | Macro-F1 | Historical verification |
|---|---|---:|---:|---:|---|
| Historical compatibility | Linear SVM | 0.5 | 82.3900% | 82.6461% | Exact at reported precision |
| Historical compatibility | Logistic Regression | 2.0 | 82.1300% | 82.4680% | Exact at reported precision |
| Clean baseline | Linear SVM | 0.5 | 82.3900% | 82.6461% | Exact at reported precision |
| Clean baseline | Logistic Regression | 2.0 | 82.1300% | 82.4680% | Exact at reported precision |

All four historical classification values were reproduced when rounded to the precision used in the original report.

## KMeans Audit

The minimal audit intentionally ran direct TF-IDF + KMeans only:

- the complete 50,000-row dataset;
- the same TF-IDF configuration used in the archived script;
- `n_clusters=10`;
- `n_init=10`;
- `random_state=42`; and
- no SVD.

The regenerated NMI was `0.18833817212523157`, compared with the historical value `0.1740`. The absolute difference was `0.014338172125231585`, so the historical KMeans metric was **not reproduced** under the direct TF-IDF interpretation.

The archived clustering script applies an ambiguous SVD path to sparse TF-IDF features, while the report describes the result simply as TF-IDF + KMeans. Because the SVD branch was explicitly excluded from Phase 1B-minimal, this audit does not determine which historical transformation produced `0.1740`.

## Additional Findings

- The archived scripts combine several representations and would run Word2Vec and Doc2Vec when executed directly; the minimal pipeline avoids those branches.
- The archived scripts write figures to existing result filenames and could overwrite Phase 1A artifacts; the new pipeline writes only to a unique run directory.
- The archived data path differs from the Phase 1A `data/raw/` policy; the new CLI accepts an explicit local path but never records that path in metadata.
- The archived plotting path depends on an untracked CJK font. Metric reproduction does not require plotting or a font file.
- Scikit-learn 1.7.2 warns that multiclass `liblinear` support will change in version 1.8. It remains unchanged here because this audit targets the historical configuration.

## Interpretation Boundary

The classification results can be described as reproduced at the precision of the original report. The KMeans result cannot yet be described as reproduced. Phase 1B-minimal does not establish end-to-end reproducibility for every method in the historical coursework project.
