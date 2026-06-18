# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from utils import setup_matplotlib_chinese, get_chinese_font

# ========================
# 数据加载
# ========================
def load_cnews(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, text = line.split("\t", maxsplit=1)
            texts.append(text)
            labels.append(label)
    return texts, labels

# ========================
# Word2Vec / Doc2Vec 特征
# ========================
def train_word2vec(texts, size=100, window=5, min_count=5):
    sentences = [text.split() for text in texts]
    w2v = Word2Vec(sentences, vector_size=size, window=window, min_count=min_count, workers=4, seed=42)
    features = []
    for s in sentences:
        vecs = [w2v.wv[w] for w in s if w in w2v.wv]
        if vecs:
            features.append(np.mean(vecs, axis=0))
        else:
            features.append(np.zeros(size))
    return np.array(features)

def train_doc2vec(texts, size=100, epochs=20):
    documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(texts)]
    d2v = Doc2Vec(documents, vector_size=size, window=5, min_count=5, workers=4, epochs=epochs, seed=42)
    features = np.array([d2v.infer_vector(text.split()) for text in texts])
    return features

# ========================
# 聚类与评测
# ========================
def cluster_evaluate(X, y_true, method_name="KMeans", **kwargs):
    if method_name == "KMeans":
        n_clusters = kwargs.get("n_clusters", len(set(y_true)))
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    elif method_name == "DBSCAN":
        eps = kwargs.get("eps", 1.5)
        min_samples = kwargs.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    else:
        raise ValueError("Unknown method name")

    y_pred = model.fit_predict(X)

    if method_name == "DBSCAN":
        valid_mask = y_pred != -1
        if valid_mask.sum() > 0:
            ari = adjusted_rand_score(y_true[valid_mask], y_pred[valid_mask])
            nmi = normalized_mutual_info_score(y_true[valid_mask], y_pred[valid_mask])
        else:
            ari, nmi = 0.0, 0.0
        noise_ratio = (y_pred == -1).mean()
    else:
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        noise_ratio = 0.0

    return ari, nmi, noise_ratio

# ========================
# 主流程
# ========================
def main():
    setup_matplotlib_chinese()
    ch_font = get_chinese_font()

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "../data/cnews.train.txt")
    results_dir = os.path.join(base_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)

    texts, labels = load_cnews(data_path)
    label_names = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(label_names)}
    y_true = np.array([label2id[l] for l in labels])

    print(f"样本数: {len(texts)}, 类别数: {len(label_names)}")

    # ========================
    # 特征探索
    # ========================
    print("生成 TF-IDF 特征...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X_tfidf = vectorizer.fit_transform(texts)

    print("训练 Word2Vec 特征...")
    X_w2v = train_word2vec(texts, size=100)

    print("训练 Doc2Vec 特征...")
    X_d2v = train_doc2vec(texts, size=100)

    features = {"TF-IDF": X_tfidf, "Word2Vec": X_w2v, "Doc2Vec": X_d2v}

    svd_dims = [50, 100, 200]
    kmeans_clusters = [len(label_names), len(label_names)+5]
    dbscan_eps = [1.0, 1.5, 2.0]

    results = []

    for feat_name, X in features.items():
        print(f"\n=== 特征: {feat_name} ===")
        # SVD 降维（仅对稀疏矩阵）
        svd_options = svd_dims if isinstance(X, np.ndarray) else [X.shape[1]]
        for dim in svd_options:
            if hasattr(X, "todense") or hasattr(X, "toarray"):
                svd = TruncatedSVD(n_components=dim, random_state=42)
                X_reduce = svd.fit_transform(X)
                print(f"SVD {dim} -> 累计方差: {svd.explained_variance_ratio_.sum():.4f}")
            else:
                X_reduce = X
                dim = X.shape[1]

            # KMeans 探索
            for k in kmeans_clusters:
                ari, nmi, _ = cluster_evaluate(X_reduce, y_true, "KMeans", n_clusters=k)
                print(f"KMeans k={k} ARI={ari:.4f} NMI={nmi:.4f}")
                results.append((feat_name, dim, f"KMeans_{k}", ari, nmi))

            # DBSCAN 探索
            for eps in dbscan_eps:
                ari, nmi, noise = cluster_evaluate(X_reduce, y_true, "DBSCAN", eps=eps, min_samples=5)
                print(f"DBSCAN eps={eps} ARI={ari:.4f} NMI={nmi:.4f} 噪声比例={noise:.2%}")
                results.append((feat_name, dim, f"DBSCAN_{eps}", ari, nmi))

    # ========================
    # 可视化：ARI / NMI 对比
    # ========================
    import pandas as pd
    df = pd.DataFrame(results, columns=["Feature", "SVD_dim", "Method", "ARI", "NMI"])
    for metric in ["ARI", "NMI"]:
        plt.figure(figsize=(12, 6))
        for feat_name in df["Feature"].unique():
            sub_df = df[df["Feature"] == feat_name]
            plt.plot(sub_df["Method"], sub_df[metric], marker='o', label=feat_name)
        plt.xticks(rotation=45, ha='right', fontproperties=ch_font)
        plt.ylabel(metric, fontproperties=ch_font)
        plt.title(f"聚类性能对比 ({metric})", fontproperties=ch_font)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"cluster_{metric}.png"), dpi=300)
        plt.close()

    print("\n✔ 聚类评估图已保存")

if __name__ == "__main__":
    main()
