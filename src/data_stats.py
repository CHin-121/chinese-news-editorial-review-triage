import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from utils import get_chinese_font, setup_matplotlib_chinese


def load_cnews(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, text = line.split("\t", maxsplit=1)
            labels.append(label)
            texts.append(text)

    return texts, labels


def main():
    # ===== 中文显示设置 =====
    setup_matplotlib_chinese()
    ch_font = get_chinese_font()

    # ===== 加载数据 =====
    texts, labels = load_cnews("../data/cnews.train.txt")

    # ===== 数据统计 =====
    total_samples = len(texts)
    label_counter = Counter(labels)
    lengths = [len(text) for text in texts]

    print(f"总样本数: {total_samples}")
    print("类别分布:")
    for label, cnt in label_counter.items():
        print(f"{label}: {cnt}")

    print(f"平均文本长度: {np.mean(lengths):.2f}")
    print(f"最大文本长度: {np.max(lengths)}")
    print(f"最小文本长度: {np.min(lengths)}")

    # ===== 确保 results 目录存在 =====
    results_dir = os.path.join(os.path.dirname(__file__), "../results")
    os.makedirs(results_dir, exist_ok=True)

    # ===== 文本长度分布可视化 =====
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50)

    plt.title("新闻文本长度分布", fontproperties=ch_font)
    plt.xlabel("文本长度", fontproperties=ch_font)
    plt.ylabel("样本数", fontproperties=ch_font)

    ax = plt.gca()
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(ch_font)

    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, "text_length_distribution.png"),
        dpi=300
    )
    plt.close()

    print("✔ 文本长度分布图已保存")


if __name__ == "__main__":
    main()






