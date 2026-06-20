"""Historical-compatible and clean TF-IDF classification tracks."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


def _split_and_encode(texts: list[str], labels: list[str], config: dict[str, Any]):
    split = config["split"]
    stratify = labels if split["stratify"] else None
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=stratify,
    )
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return X_train, X_test, y_train_encoded, y_test_encoded, encoder.classes_.tolist()


def _vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    tfidf = config["tfidf"]
    return TfidfVectorizer(
        max_features=tfidf["max_features"],
        ngram_range=tuple(tfidf["ngram_range"]),
        min_df=tfidf["min_df"],
    )


def _estimator(model_name: str, model_config: dict[str, Any], *, clean: bool):
    if model_name == "Linear SVM":
        kwargs = {"dual": model_config["dual"]}
        if clean:
            kwargs["random_state"] = model_config["random_state"]
        return LinearSVC(**kwargs)
    if model_name == "Logistic Regression":
        kwargs = {
            "solver": model_config["solver"],
            "max_iter": model_config["max_iter"],
        }
        if clean:
            kwargs["random_state"] = model_config["random_state"]
        return LogisticRegression(**kwargs)
    raise ValueError(f"Unsupported model: {model_name}")


def _metric_rows(
    *,
    result_type: str,
    model_name: str,
    accuracy: float,
    macro_f1: float,
    best_params: dict[str, Any],
) -> list[dict[str, Any]]:
    common = {
        "result_type": result_type,
        "task": "classification",
        "representation": "TF-IDF",
        "model": model_name,
        "unit": "percent",
        "best_params": best_params,
        "source": "Phase 1B-minimal regenerated run",
    }
    return [
        {**common, "metric": "Accuracy", "value": accuracy * 100.0},
        {**common, "metric": "Macro-F1", "value": macro_f1 * 100.0},
    ]


def run_historical_compat(
    texts: list[str], labels: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, classes = _split_and_encode(texts, labels, config)
    vectorizer = _vectorizer(config)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    cv = config["cross_validation"]
    rows: list[dict[str, Any]] = []

    for model_name, model_config in config["models"].items():
        grid = GridSearchCV(
            _estimator(model_name, model_config, clean=False),
            {"C": model_config["C"]},
            cv=cv["folds"],
            scoring=cv["scoring"],
            n_jobs=cv["n_jobs"],
        )
        grid.fit(X_train_tfidf, y_train)
        prediction = grid.best_estimator_.predict(X_test_tfidf)
        rows.extend(
            _metric_rows(
                result_type=config["result_type"],
                model_name=model_name,
                accuracy=accuracy_score(y_test, prediction),
                macro_f1=f1_score(y_test, prediction, average="macro"),
                best_params=grid.best_params_,
            )
        )

    return {
        "metrics": rows,
        "details": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "class_labels": classes,
            "feature_count": X_train_tfidf.shape[1],
            "tfidf_fit_scope": "complete outer training split before cross-validation",
        },
    }


def run_clean_baseline(
    texts: list[str], labels: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test, classes = _split_and_encode(texts, labels, config)
    cv = config["cross_validation"]
    rows: list[dict[str, Any]] = []

    for model_name, model_config in config["models"].items():
        pipeline = Pipeline(
            [
                ("tfidf", _vectorizer(config)),
                ("classifier", _estimator(model_name, model_config, clean=True)),
            ]
        )
        grid = GridSearchCV(
            pipeline,
            {"classifier__C": model_config["C"]},
            cv=cv["folds"],
            scoring=cv["scoring"],
            n_jobs=cv["n_jobs"],
        )
        grid.fit(X_train, y_train)
        prediction = grid.best_estimator_.predict(X_test)
        best_params = {"C": grid.best_params_["classifier__C"]}
        rows.extend(
            _metric_rows(
                result_type=config["result_type"],
                model_name=model_name,
                accuracy=accuracy_score(y_test, prediction),
                macro_f1=f1_score(y_test, prediction, average="macro"),
                best_params=best_params,
            )
        )

    return {
        "metrics": rows,
        "details": {
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "class_labels": classes,
            "tfidf_fit_scope": "inside each cross-validation fold via sklearn Pipeline",
        },
    }
