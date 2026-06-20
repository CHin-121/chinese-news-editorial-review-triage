# Chinese News Auto-Categorization and Editorial Review Triage

## Practical Problem

A news desk may use automatic categorization to route incoming Chinese articles into editorial sections, while sending uncertain cases to human editors. The useful question is therefore not only how often the classifier is correct, but also which categories are difficult and whether a limited review budget can concentrate on likely mistakes.

This Phase 1C-minimal analysis uses aggregate test-set outputs only. It does not store article text, predictions tied to individual articles, private paths, or user-behavior data.

## Model Used

The analysis reuses the Phase 1B clean baseline:

- TF-IDF with `max_features=5000`, `ngram_range=(1, 2)`, and `min_df=5`;
- an 80/20 stratified split with `random_state=42`; and
- `LinearSVC(dual=False, C=0.5, random_state=42)`.

The selected `C=0.5` comes from the earlier clean-baseline cross-validation. The final pipeline was fitted once on 40,000 training records and evaluated on the unchanged 10,000-record test split. It achieved 82.39% accuracy and 82.65% Macro-F1 after rounding to two decimal places.

## Why These Diagnostics Are Used

The per-class report shows whether overall accuracy hides uneven category performance. The confusion matrix identifies which true sections are systematically routed to other sections. The sorted confusion-pair table turns those cells into an editorially readable error list.

For review triage, the Linear SVM decision scores are ranked using:

```text
review_margin = top1_score - top2_score
```

A smaller margin means the two leading class scores are closer, so the item receives higher review priority. This margin is a ranking signal, not a probability or calibrated confidence estimate.

## Key Findings

The strongest per-class F1 scores were for 房产 (0.9684), 体育 (0.9493), and 娱乐 (0.9404). The weakest was 家居 (0.6589), followed by 教育 (0.7145). This unevenness matters because a single overall accuracy value would otherwise obscure the substantially lower reliability of some editorial sections.

The largest directional confusion was 家居 predicted as 时尚, with 163 errors, representing 9.26% of all test errors. The next two were 时尚 predicted as 家居 (117 errors) and 时政 predicted as 家居 (108 errors). These patterns suggest that 家居 acts as an important overlap point among lifestyle and general-news vocabulary in this dataset.

The margin-based review policy produced the following aggregate results:

| Review budget | Reviewed | Errors captured | Share of all errors | Errors per 100 reviewed |
|---:|---:|---:|---:|---:|
| 5% | 500 | 326 | 18.51% | 65.2 |
| 10% | 1,000 | 629 | 35.72% | 62.9 |
| 20% | 2,000 | 1,058 | 60.08% | 52.9 |

The ranking therefore concentrates errors substantially above the overall test error rate of 17.61%. The result supports using the decision margin as a simple editorial review queue, while leaving high-margin items to the automatic routing workflow.

## Limitations

- The evaluation uses a historical, balanced cnews dataset and does not establish performance on a current newsroom stream.
- Decision margins are not probabilities and are not calibrated across models or deployments.
- Low-margin items are not guaranteed to be incorrect, and high-margin items can still contain errors.
- The analysis evaluates category routing only; it does not measure editorial quality, business impact, CTR, recommendations, or user behavior.
- No external data, deep learning, BERT, Word2Vec, or Doc2Vec was used.
- The outputs are aggregate diagnostics. They do not include raw text or a per-article review queue.
