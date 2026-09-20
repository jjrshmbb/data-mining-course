"""全局配置: 路径、随机种子、目标列等。

所有模块统一从这里取值, 避免魔法数字散落各处。
换数据集时, 通常只需要改 TARGET_COLUMN / DROP_COLUMNS。
"""

from __future__ import annotations

from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_FIGURES = REPORTS / "figures"

# ---------- 实验设置 ----------
RANDOM_SEED = 42          # 固定随机种子, 保证结果可复现
TEST_SIZE = 0.2           # 测试集比例
CV_FOLDS = 5              # 交叉验证折数

# ---------- 数据设置（换数据集时改这里）----------
TARGET_COLUMN = "target"  # TODO: 改成你数据集的目标列名
DROP_COLUMNS: list[str] = []   # TODO: 需要丢弃的列, 例如 ID 列、会造成标签泄漏的列
POSITIVE_LABEL = None     # 二分类可指定正类标签(影响 precision/recall), None 表示自动

# 自动创建目录, 保证首次运行不会因缺目录报错
for _path in (DATA_RAW, DATA_PROCESSED, REPORTS, REPORTS_FIGURES):
    _path.mkdir(parents=True, exist_ok=True)
