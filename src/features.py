"""特征工程: 衍生特征构造 + 特征筛选。

脚手架默认不改动特征(保证流程先跑通), 只做通用的安全筛选(常数列、高度冗余列)。
针对你自己数据集的业务特征是最容易拿高分的地方, 在 build_features 里按示例扩展。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_ratio_feature(X_train: pd.DataFrame, X_test: pd.DataFrame,
                      numerator: str, denominator: str,
                      name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造比值特征, 例如 面积/房间数 = 户均面积。"""
    name = name or f"{numerator}_per_{denominator}"
    for X in (X_train, X_test):
        if numerator in X.columns and denominator in X.columns:
            X[name] = X[numerator] / X[denominator].replace(0, np.nan)
            X[name] = X[name].fillna(X[name].median() if X[name].notna().any() else 0.0)
    return X_train, X_test


def add_interaction(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    col_a: str, col_b: str, name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造交互特征(乘积), 例如 价格*数量 = 总额。"""
    name = name or f"{col_a}_x_{col_b}"
    for X in (X_train, X_test):
        if col_a in X.columns and col_b in X.columns:
            X[name] = X[col_a] * X[col_b]
    return X_train, X_test


def add_bin_feature(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    column: str, bins: int = 5, name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造分箱特征(等频分箱), 例如把年龄分成 5 档。"""
    name = name or f"{column}_bin"
    if column not in X_train.columns:
        return X_train, X_test
    try:
        X_train[name], edges = pd.qcut(X_train[column], q=bins, labels=False, retbins=True, duplicates="drop")
        X_test[name] = pd.cut(X_test[column], bins=edges, labels=False, include_lowest=True)
        X_test[name] = X_test[name].fillna(-1)
    except ValueError:
        return X_train, X_test
    return X_train, X_test


def build_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """本项目定制特征的入口。默认原样返回, 保证流水线先跑通。

    示例(取消注释, 并把列名换成你数据集里的列名):

        X_train, X_test = add_ratio_feature(X_train, X_test, "total_area", "rooms")
        X_train, X_test = add_interaction(X_train, X_test, "price", "quantity")
        X_train, X_test = add_bin_feature(X_train, X_test, "age", bins=5)
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # TODO: 在这里加入你的业务特征

    return X_train, X_test


def select_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series | None = None,
                    corr_threshold: float = 0.95) -> tuple[pd.DataFrame, pd.DataFrame]:
    """特征筛选: 1) 去掉常数特征; 2) 去掉高度相关的冗余特征。"""
    # 1) 常数列(方差为 0), 对模型没有信息量
    nunique = X_train.nunique()
    keep = nunique[nunique > 1].index.tolist()
    dropped_const = [c for c in X_train.columns if c not in keep]

    # 2) 相关性过高的特征只保留一个(优先保留与目标相关性更高的)
    if len(keep) > 1:
        corr = X_train[keep].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        target_corr = None
        if y_train is not None and pd.api.types.is_numeric_dtype(y_train):
            try:
                target_corr = X_train[keep].corrwith(y_train).abs()
            except Exception:
                target_corr = None
        dropped_corr = []
        for col in upper.columns:
            partners = upper.index[upper[col] > corr_threshold].tolist()
            if not partners or col in dropped_corr:
                continue
            for other in partners:
                if other in dropped_corr:
                    continue
                if target_corr is not None:
                    loser = other if target_corr.get(other, 0) < target_corr.get(col, 0) else col
                else:
                    loser = other
                dropped_corr.append(loser)
        keep = [c for c in keep if c not in set(dropped_corr)]
    else:
        dropped_corr = []

    dropped = sorted(set(dropped_const + dropped_corr))
    if dropped:
        print(f"      特征筛选: 去掉 {len(dropped)} 个特征 -> {dropped[:10]}{' ...' if len(dropped) > 10 else ''}")

    return X_train[keep], X_test[keep]
