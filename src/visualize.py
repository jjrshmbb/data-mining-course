"""可视化: EDA 图、模型对比、混淆矩阵。

使用 Agg 后端, 即使没有图形界面(如远程服务器)也能正常出图并保存到 reports/figures/。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")   # 必须在 pyplot 之前设置

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402
import seaborn as sns                    # noqa: E402
from matplotlib import font_manager      # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from . import config                     # noqa: E402


def setup_chinese_font() -> str | None:
    """自动挑选可用的中文字体, 避免图表中文显示为方框。"""
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
    for name in candidates:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _save(fig, filename: str) -> None:
    path = config.REPORTS_FIGURES / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      图表已保存: {path}")


def plot_eda(df: pd.DataFrame, target: str) -> None:
    """探索性分析: 目标变量分布、相关性热力图、数值特征分布。"""
    setup_chinese_font()
    sns.set_theme(style="whitegrid")

    # 1) 目标变量分布
    fig, ax = plt.subplots(figsize=(7, 4))
    if df[target].nunique() <= 20:
        sns.countplot(x=df[target], ax=ax)
        ax.set_title(f"目标变量分布: {target}")
    else:
        sns.histplot(df[target].dropna(), kde=True, ax=ax)
        ax.set_title(f"目标变量分布(直方图): {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("样本数")
    _save(fig, "01_target_distribution.png")

    # 2) 数值特征相关性热力图
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(max(6, numeric.shape[1] * 0.6), max(5, numeric.shape[1] * 0.5)))
        sns.heatmap(numeric.corr(), annot=numeric.shape[1] <= 12, fmt=".2f",
                    cmap="coolwarm", center=0, square=False, ax=ax)
        ax.set_title("数值特征相关性热力图")
        _save(fig, "02_correlation_heatmap.png")

    # 3) 数值特征分布(最多取前 6 个)
    cols = [c for c in numeric.columns if c != target][:6]
    if cols:
        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"{col} 分布")
        for ax in axes[len(cols):]:
            ax.axis("off")
        _save(fig, "03_feature_distributions.png")


def plot_model_comparison(table: pd.DataFrame) -> None:
    """模型对比柱状图(按主指标)。"""
    setup_chinese_font()
    sns.set_theme(style="whitegrid")
    main = "accuracy" if "accuracy" in table.columns else "r2"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=table, x=main, y="model", ax=ax, palette="viridis", hue="model", legend=False)
    for i, v in enumerate(table[main]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center", fontsize=9)
    ax.set_title(f"模型对比(测试集 {main})")
    ax.set_xlim(0, min(1.05, float(table[main].max()) * 1.15) if main == "accuracy" else None)
    _save(fig, "04_model_comparison.png")

    # 交叉验证成绩
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(table["model"], table["cv_mean"], xerr=table["cv_std"], color="#4C72B0", alpha=0.85)
    ax.set_title(f"交叉验证平均成绩({config.CV_FOLDS} 折)")
    ax.set_xlabel("得分")
    ax.invert_yaxis()
    _save(fig, "05_cross_validation.png")


def plot_confusion_matrices(results: dict, y_test) -> None:
    """为每个分类模型绘制混淆矩阵。"""
    setup_chinese_font()
    sns.set_theme(style="white")

    labels = sorted(pd.unique(y_test).tolist(), key=str)
    for name, r in results.items():
        if r["task"] != "classification":
            continue
        cm = confusion_matrix(y_test, r["y_pred"], labels=labels)
        fig, ax = plt.subplots(figsize=(max(4.5, len(labels) * 0.9), max(4, len(labels) * 0.8)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f"混淆矩阵 - {name}")
        ax.set_xlabel("预测值")
        ax.set_ylabel("真实值")
        _save(fig, f"cm_{name}.png")


def plot_feature_importance(model, feature_names, top_n: int = 15, name: str = "model") -> None:
    """绘制树模型的特征重要性(可选调用)。"""
    setup_chinese_font()
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(3, len(importances) * 0.35)))
    importances[::-1].plot(kind="barh", ax=ax, color="#55A868")
    ax.set_title(f"特征重要性 Top{len(importances)} - {name}")
    ax.set_xlabel("重要性")
    _save(fig, f"06_feature_importance_{name}.png")
