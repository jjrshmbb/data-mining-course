"""一键跑通数据挖掘完整流程: 读取 -> 预处理 -> 特征工程 -> 建模 -> 评估。

用法:
    python main.py --demo                      # 用内置演示数据跑通流程
    python main.py --data data/raw/xxx.csv     # 用真实数据
    python main.py --data data/raw/xxx.csv --model random_forest
"""

from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.data_loader import load_data
from src.preprocess import preprocess
from src.features import build_features, select_features
from src.train import train_models
from src.evaluate import evaluate_models, save_report
from src.visualize import plot_eda, plot_confusion_matrices, plot_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="数据挖掘课程作业 - 主流程")
    parser.add_argument("--data", type=str, default=None,
                        help="数据文件路径(csv/excel); 不传则用 --demo 的演示数据")
    parser.add_argument("--demo", action="store_true",
                        help="使用 sklearn 自带 iris 数据集演示")
    parser.add_argument("--model", type=str, default="all",
                        help="指定单个模型名, 默认 all 表示全部对比")
    parser.add_argument("--target", type=str, default=None,
                        help="目标列名, 默认取 config.TARGET_COLUMN")
    parser.add_argument("--no-plots", action="store_true", help="跳过绘图(服务器上更快)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.target or config.TARGET_COLUMN
    t0 = time.time()

    print("=" * 64)
    print("数据挖掘课程作业 - 主流程")
    print("=" * 64)
    print(f"[配置] 随机种子={config.RANDOM_SEED}  测试集比例={config.TEST_SIZE}  目标列={target}")

    # 1. 读取数据
    print("\n[1/5] 读取数据 ...")
    df, source = load_data(args.data, demo=args.demo or args.data is None)
    print(f"      数据来源: {source}")
    print(f"      数据规模: {df.shape[0]} 行 x {df.shape[1]} 列")
    if target not in df.columns and target.lower() not in [c.lower() for c in df.columns]:
        print(f"[错误] 目标列 '{target}' 不存在。现有列: {list(df.columns)}")
        print("       请用 --target 指定正确的目标列 (或在 src/config.py 中修改 TARGET_COLUMN)。")
        return 2

    # 2. 探索性分析图
    if not args.no_plots:
        print("\n[2/5] 生成探索性分析图表 ...")
        plot_eda(df, target)

    # 3. 预处理
    print("\n[3/5] 数据预处理 ...")
    X_train, X_test, y_train, y_test, meta = preprocess(df, target)
    print(f"      训练集 {X_train.shape}, 测试集 {X_test.shape}")

    # 4. 特征工程
    print("\n[4/5] 特征工程 ...")
    X_train, X_test = build_features(X_train, X_test)
    X_train, X_test = select_features(X_train, X_test, y_train)

    # 5. 建模与评估
    print("\n[5/5] 模型训练与评估 ...")
    results = train_models(X_train, y_train, X_test, y_test, only=None if args.model == "all" else args.model)
    table = evaluate_models(results, y_test)
    print("\n" + table.to_string(index=False))

    best = table.iloc[0]["model"]
    print(f"\n[结论] 表现最好的模型: {best}")

    save_report(table, results, y_test, target, source, meta)
    if not args.no_plots:
        plot_model_comparison(table)
        plot_confusion_matrices(results, y_test)

    print(f"\n完成, 耗时 {time.time() - t0:.1f} 秒。报告见 {config.REPORTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
