"""数据读取。

优先读取 data/raw/ 下的数据文件; 若没有, 回退到 sklearn 自带数据集,
保证脚手架"开箱即可运行"（先跑通流程, 再换成真实数据）。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

SUPPORTED_SUFFIXES = {".csv", ".txt", ".xlsx", ".xls", ".parquet"}


def list_raw_files() -> list[Path]:
    """列出 data/raw/ 下受支持的数据文件。"""
    if not config.DATA_RAW.exists():
        return []
    return sorted(
        p for p in config.DATA_RAW.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith(".")
    )


def read_file(path: str | Path) -> pd.DataFrame:
    """按后缀读取数据文件; csv 会自动尝试 utf-8 / gbk 两种编码。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在: {p}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(p, encoding="gbk")   # 中文数据集常见编码
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".txt":
        return pd.read_csv(p, sep=None, engine="python")   # 自动推断分隔符
    raise ValueError(f"暂不支持的文件类型: {suffix}")


def load_demo() -> pd.DataFrame:
    """sklearn 自带 iris 数据集, 用于演示与自测。"""
    from sklearn.datasets import load_iris

    bunch = load_iris(as_frame=True)
    df = bunch.frame.copy()
    # 列名规范化: "sepal length (cm)" -> "sepal_length"
    df.columns = [str(c).replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    return df


def load_data(path: str | Path | None = None, demo: bool = False) -> tuple[pd.DataFrame, str]:
    """加载数据, 返回 (DataFrame, 数据来源说明)。"""
    if demo:
        return load_demo(), "sklearn iris 演示数据 (--demo)"
    if path:
        return read_file(path), str(path)

    files = list_raw_files()
    if files:
        return read_file(files[0]), str(files[0])
    return load_demo(), "sklearn iris 演示数据 (未在 data/raw/ 找到数据文件)"
