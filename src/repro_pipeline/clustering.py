"""Direct TF-IDF plus KMeans baseline without the historical SVD branch."""

from __future__ import annotations

from typing import Any

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


def run_direct_tfidf_kmeans(
    texts: list[str], labels: list[str], config: dict[str, Any]
) -> dict[str, Any]:
    tfidf = config["tfidf"]
    vectorizer = TfidfVectorizer(
        max_features=tfidf["max_features"],
        ngram_range=tuple(tfidf["ngram_range"]),
        min_df=tfidf["min_df"],
    )
    features = vectorizer.fit_transform(texts)
    encoded_labels = LabelEncoder().fit_transform(labels)
    kmeans_config = config["kmeans"]
    model = KMeans(
        n_clusters=kmeans_config["n_clusters"],
        n_init=kmeans_config["n_init"],
        random_state=kmeans_config["random_state"],
    )
    predicted_clusters = model.fit_predict(features)
    nmi = normalized_mutual_info_score(encoded_labels, predicted_clusters)
    metric = {
        "result_type": kmeans_config["result_type"],
        "task": "clustering",
        "representation": "TF-IDF",
        "model": "KMeans",
        "metric": "NMI",
        "value": nmi,
        "unit": "score",
        "best_params": {
            "n_clusters": kmeans_config["n_clusters"],
            "n_init": kmeans_config["n_init"],
            "random_state": kmeans_config["random_state"],
        },
        "source": "Phase 1B-minimal regenerated run",
    }
    return {
        "metrics": [metric],
        "details": {
            "sample_count": features.shape[0],
            "feature_count": features.shape[1],
            "n_clusters": kmeans_config["n_clusters"],
            "svd_used": False,
        },
    }
