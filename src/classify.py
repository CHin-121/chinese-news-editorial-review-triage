# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

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
    # 平均每句话的词向量
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
# 模型训练与评测
# ========================
def train_evaluate(X_train, y_train, X_test, y_test, model_name="SVM"):
    if model_name == "SVM":
        clf = LinearSVC(dual=False)
        param_grid = {"C": [0.1, 0.5, 1.0, 2.0]}
    elif model_name == "NB":
        clf = MultinomialNB()
        param_grid = {"alpha": [0.1, 0.5, 1.0]}
    elif model_name == "LR":
        clf = LogisticRegression(max_iter=500, solver="liblinear")
        param_grid = {"C": [0.1, 0.5, 1.0, 2.0]}
    else:
        raise ValueError("Unknown model name")

    grid = GridSearchCV(clf, param_grid, cv=5, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    return acc, f1, grid.best_params_

# ========================
# 主流程
# ========================
def main():
    setup_matplotlib_chinese()
    ch_font = get_chinese_font()

    base_dir = os.path.dirname(__file__)
    train_path = os.path.join(base_dir, "../data/cnews.train.txt")
    results_dir = os.path.join(base_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)

    # 只有 train 数据，拆分出 train / test 20%
    texts, labels = load_cnews(train_path)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # ========================
    # 1️⃣ TF-IDF 特征
    # ========================
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 分类评测
    acc_svm, f1_svm, best_svm = train_evaluate(X_train_tfidf, y_train_enc, X_test_tfidf, y_test_enc, "SVM")
    acc_nb, f1_nb, best_nb = train_evaluate(X_train_tfidf, y_train_enc, X_test_tfidf, y_test_enc, "NB")
    acc_lr, f1_lr, best_lr = train_evaluate(X_train_tfidf, y_train_enc, X_test_tfidf, y_test_enc, "LR")

    print("\n=== TF-IDF 特征分类结果 ===")
    print(f"SVM: Accuracy={acc_svm:.4f}, F1-macro={f1_svm:.4f}, best_params={best_svm}")
    print(f"NB : Accuracy={acc_nb:.4f}, F1-macro={f1_nb:.4f}, best_params={best_nb}")
    print(f"LR : Accuracy={acc_lr:.4f}, F1-macro={f1_lr:.4f}, best_params={best_lr}")

    # ========================
    # 2️⃣ Word2Vec 特征
    # ========================
    print("\n训练 Word2Vec 特征...")
    X_train_w2v = train_word2vec(X_train, size=100)
    X_test_w2v = train_word2vec(X_test, size=100)

    acc_svm_w2v, f1_svm_w2v, _ = train_evaluate(X_train_w2v, y_train_enc, X_test_w2v, y_test_enc, "SVM")
    print(f"Word2Vec SVM: Accuracy={acc_svm_w2v:.4f}, F1-macro={f1_svm_w2v:.4f}")

    # ========================
    # 3️⃣ Doc2Vec 特征
    # ========================
    print("\n训练 Doc2Vec 特征...")
    X_train_d2v = train_doc2vec(X_train, size=100)
    X_test_d2v = train_doc2vec(X_test, size=100)
    acc_svm_d2v, f1_svm_d2v, _ = train_evaluate(X_train_d2v, y_train_enc, X_test_d2v, y_test_enc, "SVM")
    print(f"Doc2Vec SVM: Accuracy={acc_svm_d2v:.4f}, F1-macro={f1_svm_d2v:.4f}")

    # ========================
    # 可视化对比
    # ========================
    models = ["SVM-TFIDF", "NB-TFIDF", "LR-TFIDF", "SVM-Word2Vec", "SVM-Doc2Vec"]
    acc_scores = [acc_svm, acc_nb, acc_lr, acc_svm_w2v, acc_svm_d2v]
    f1_scores = [f1_svm, f1_nb, f1_lr, f1_svm_w2v, f1_svm_d2v]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, acc_scores, width, label="Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="F1-macro")
    plt.xticks(x, models, fontproperties=ch_font, rotation=30)
    plt.ylabel("得分", fontproperties=ch_font)
    plt.title("文本分类模型性能对比", fontproperties=ch_font)
    plt.legend()

    ax = plt.gca()
    for label in ax.get_yticklabels():
        label.set_fontproperties(ch_font)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "classification_comparison.png"), dpi=300)
    plt.close()

    print("\n✔ 分类对比图已保存")

if __name__ == "__main__":
    main()
