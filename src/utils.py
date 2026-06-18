# utils.py
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def get_chinese_font():
    """
    返回项目内置的中文字体 FontProperties
    """
    font_path = os.path.join(
        os.path.dirname(__file__),
        "../fonts/NotoSerifCJKsc-Light.otf"
    )

    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")

    return FontProperties(fname=font_path)

def setup_matplotlib_chinese():
    """
    解决 matplotlib 中文显示 & 负号问题
    """
    plt.rcParams["axes.unicode_minus"] = False
