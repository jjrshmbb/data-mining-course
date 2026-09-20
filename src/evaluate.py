"""模型评估: 指标计算、模型对比表、评估报告导出。"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)

from . import config


def evaluate_models(results: dict, y_test) -> pd.DataFrame:
    """把各模型的测试集指标汇总成一张对比表(按主指标降序)。"""
    rows = []
    for name, r in results.items():
        row = {"model": name, "cv_mean": round(r["cv_mean"], 4), "cv_std": round(r["cv_std"], 4)}
        if r["task"] == "classification":
            y_pred = r["y_pred"]
            row["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            row["precision"] = round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4)
            row["recall"] = round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4)
            row["f1"] = round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)
            row["roc_auc"] = _safe_auc(y_test, r.get("y_score"))
        else:
            row["r2"] = round(r2_score(y_test, r["y_pred"]), 4)
            row["mae"] = round(mean_absolute_error(y_test, r["y_pred"]), 4)
            row["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, r["y_pred"]))), 4)
        rows.append(row)

    table = pd.DataFrame(rows)
    main = "accuracy" if "accuracy" in table.columns else "r2"
    return table.sort_values(main, ascending=False).reset_index(drop=True)


def _safe_auc(y_test, y_score):
    """计算 ROC-AUC; 类别或分数不满足条件时返回 NaN。"""
    if y_score is None:
        return np.nan
    try:
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            return round(roc_auc_score(y_test, y_score[:, 1]), 4)
        if y_score.ndim == 1:
            return round(roc_auc_score(y_test, y_score), 4)
        return round(roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted"), 4)
    except Exception:
        return np.nan


def save_report(table: pd.DataFrame, results: dict, y_test, target: str,
                source: str, meta=None) -> object:
    """导出 Markdown 评估报告到 reports/。"""
    path = config.REPORTS / "evaluation_report.md"
    best = table.iloc[0]["model"]
    task = results[best]["task"]

    lines = [
        "# 模型评估报告",
        "",
        f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据来源: {source}",
        f"- 目标列: {target}",
        f"- 任务类型: {'分类' if task == 'classification' else '回归'}",
        f"- 随机种子: {config.RANDOM_SEED}   测试集比例: {config.TEST_SIZE}   交叉验证折数: {config.CV_FOLDS}",
        "",
        "## 一、模型对比",
        "",
        table.to_markdown(index=False),
        "",
        f"**最优模型: {best}**",
        "",
    ]

    if meta is not None:
        lines += [
            "## 二、数据与预处理记录",
            "",
            f"- 样本数: {meta.n_rows_before} -> {meta.n_rows_after} (删除重复 {meta.n_duplicates_removed} 行)",
            f"- 数值特征 {len(meta.numeric_columns)} 个 / 类别特征 {len(meta.categorical_columns)} 个",
            f"- 丢弃列: {meta.dropped_columns or '无'}",
            "",
            "### 缺失值填充",
            "",
        ]
        if meta.missing_filled:
            lines.append("| 列名 | 缺失数 | 填充值 |")
            lines.append("|---|---|---|")
            for col, info in meta.missing_filled.items():
                lines.append(f"| {col} | {info['count']} | {info['filled_with']} |")
        else:
            lines.append("无缺失值。")
        lines += ["", "### 异常值处理(IQR 盖帽)", ""]
        if meta.outliers_clipped:
            lines.append("| 列名 | 处理数量 |")
            lines.append("|---|---|")
            for col, n in meta.outliers_clipped.items():
                lines.append(f"| {col} | {n} |")
        else:
            lines.append("未检出异常值。")
        lines.append("")

    lines += [
        "## 三、结论与讨论",
        "",
        "> 结合指标与业务含义撰写: 哪个模型最好、为什么、有哪些局限、后续如何改进。",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"      评估报告已写入: {path}")

    # 同时保存一份 CSV 便于粘贴到报告/PPT
    csv_path = config.REPORTS / "model_comparison.csv"
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"      对比表已写入: {csv_path}")
    return path


def confusion(y_test, y_pred):
    """返回混淆矩阵(供可视化使用)。"""
    return confusion_matrix(y_test, y_pred)
