"""研究三第一步：分数塌缩诊断。

只读取已有的 predictions/*.csv（每个测试样本的异常分数），不重新训练任何模型。
目的：检查"AP 没下降"的配置里，异常分数的分布是否已经改变。

三个诊断量：
1. std        —— 分数的标准差（离散度）
2. rel_std    —— 相对离散度 = 当前 std / 该流程在 0% 缺失时的 std
3. uniq_ratio —— 不同分数值的占比（出现大量并列说明分数在塌缩）
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from research.config import ROOT


def newest_run_with_all_rates(need_last_rate='70'):
    """找出最新的、包含全部缺失率的正式运行目录。"""
    latest = (ROOT / 'reports/latest_run.txt').read_text(encoding='utf-8').strip()
    cand = ROOT / latest
    if (cand / 'predictions').exists() and any((cand / 'predictions').glob(f'*rate_{need_last_rate}_*.csv')):
        return cand
    runs = sorted((ROOT / 'reports/runs').iterdir(), key=lambda p: p.name, reverse=True)
    for r in runs:
        if (r / 'predictions').exists() and any((r / 'predictions').glob(f'*rate_{need_last_rate}_*.csv')):
            return r
    raise FileNotFoundError('找不到包含 rate_70 的运行目录')


def read_scores(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return np.array([float(r['anomaly_score']) for r in csv.DictReader(f)])


def main():
    run = newest_run_with_all_rates()
    out = ROOT / 'reports' / 'score_collapse' / run.name
    out.mkdir(parents=True, exist_ok=True)
    print('分析目录:', run)

    rows = []
    for p in sorted((run / 'predictions').glob('*.csv')):
        stem = p.stem                      # seed_42_rate_00_median_iforest
        parts = stem.split('_')
        seed = int(parts[1])
        rate = int(parts[3]) / 100.0
        imputer = parts[4]
        detector = '_'.join(parts[5:])
        s = read_scores(p)
        rows.append({'seed': seed, 'missing_rate': rate,
                     'imputer': imputer, 'detector': detector,
                     'score_std': float(s.std()),
                     'score_iqr': float(np.percentile(s, 75) - np.percentile(s, 25)),
                     'uniq_ratio': len(np.unique(np.round(s, 12))) / len(s)})
    stats = pd.DataFrame(rows)

    res = pd.read_csv(run / 'results.csv', encoding='utf-8-sig')
    res = res[['seed', 'missing_rate', 'imputer', 'detector', 'AP', 'ROC_AUC']]
    df = stats.merge(res, on=['seed', 'missing_rate', 'imputer', 'detector'])

    key = ['imputer', 'detector']
    # 每个种子用自己 0% 时的离散度做基准，避免跨种子混算
    base = df[df.missing_rate == 0][['seed'] + key + ['score_std']].rename(
        columns={'score_std': 'std_at_0'})
    df = df.merge(base, on=['seed'] + key, how='left')
    df['rel_std'] = df.score_std / df.std_at_0
    df.to_csv(out / 'per_config.csv', index=False, encoding='utf-8-sig')

    agg = df.groupby(key + ['missing_rate']).agg(
        AP=('AP', 'mean'), score_std=('score_std', 'mean'),
        rel_std=('rel_std', 'mean'), uniq_ratio=('uniq_ratio', 'mean'),
        n=('AP', 'size')).reset_index()
    agg.to_csv(out / 'summary.csv', index=False, encoding='utf-8-sig')
    print(agg.round(4).to_string(index=False))

    # 画图：左 AP（绝对），右 相对离散度（相对 0%）
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'median_iforest': '#cd5c24', 'knn_iforest': '#125d98',
              'median_lof': '#8c4d9d', 'knn_lof': '#008a76'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for (imp, det), g in agg.groupby(key):
        c = colors.get(f'{imp}_{det}', '#555555')
        lab = f'{imp} + {det}'
        axes[0].plot(g.missing_rate * 100, g.AP, marker='o', color=c, linewidth=1.8, label=lab)
        axes[1].plot(g.missing_rate * 100, g.rel_std, marker='o', color=c, linewidth=1.8, label=lab)
    axes[0].set_ylabel('Average Precision (AP)')
    axes[0].set_title('AP: looks flat')
    axes[0].axhline(0.5, color='#777777', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Score spread, relative to 0% missing')
    axes[1].set_title('Score spread: not flat')
    axes[1].axhline(1.0, color='#777777', linestyle='--', linewidth=1)
    for ax in axes:
        ax.set_xlabel('MCAR missing entries (%)')
        ax.set_xticks(sorted(agg.missing_rate.unique() * 100))
        ax.grid(alpha=.2)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Same predictions, two very different stories', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / 'collapse.png', dpi=180)
    plt.close(fig)

    # 给汇报用的一句话结论
    lines = ['# 分数塌缩诊断', '',
             f'数据来源：{run.name}（{len(df)} 个配置），只读取预测分数，未重新训练。', '']
    for (imp, det), g in agg.groupby(key):
        first = g[g.missing_rate == 0].AP.iloc[0]
        last_row = g.iloc[-1]
        lines.append(f'- {imp} + {det}：AP {first:.4f} → {last_row.AP:.4f}（{last_row.AP-first:+.4f}）；'
                     f'分数离散度相对值 {last_row.rel_std:.2f}；并列比例 {last_row.uniq_ratio:.2f}')
    lines += ['', '判读：若 AP 变化接近 0 而 rel_std 明显偏离 1，说明指标没有报警，但模型输出分布已经改变。']
    (out / 'conclusion.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n输出:', out)


if __name__ == '__main__':
    main()
