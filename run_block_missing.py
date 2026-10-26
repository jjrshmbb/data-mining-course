"""研究二：集中缺失 vs 分散缺失（缺失总量精确匹配的对照实验）。

设计要点（全部来自 docs/research_protocol.md 的预先登记）：
1. 两种布局的缺失单元总数完全相同，只有分布不同。
2. 集中布局：先用与标签无关的随机种子选出一半的列，再把全部缺失塞进这些列。
3. 选列只用随机数，不看标签、不看数值大小。
4. 本脚本只新增文件，不修改 research/ 下任何既有代码。
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from research.config import DEFAULT, ROOT
from research.data import load, split_indices
from research.experiment import fit_score

SEEDS = DEFAULT['seeds']
RATES = [0.1, 0.3]
IMPUTERS = DEFAULT['imputers']
DETECTORS = DEFAULT['detectors']
BLOCK_FRACTION = 0.5           # 一半的列承受全部缺失


def total_cells(n_rows, n_cols, rate):
    """按比例算出的缺失单元总数（两种布局共用，保证精确匹配）。"""
    return int(round(rate * n_rows * n_cols))


def mask_scattered(n_rows, n_cols, rate, rng):
    """分散：在整个矩阵里随机挑 n 个格子。"""
    n = total_cells(n_rows, n_cols, rate)
    flat = rng.permutation(n_rows * n_cols)[:n]
    mask = np.zeros(n_rows * n_cols, dtype=bool)
    mask[flat] = True
    return mask.reshape(n_rows, n_cols)


def mask_block(n_rows, n_cols, rate, rng, frac=BLOCK_FRACTION):
    """集中：先随机选 frac 比例的列，再把同样多的缺失全部塞进这些列。"""
    n = total_cells(n_rows, n_cols, rate)
    n_sel = max(1, int(round(n_cols * frac)))
    cols = rng.permutation(n_cols)[:n_sel]
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    per_col, rem = divmod(n, n_sel)
    for i, c in enumerate(cols):
        k = per_col + (1 if i < rem else 0)
        if k <= 0:
            continue
        rows = rng.permutation(n_rows)[:k]
        mask[rows, c] = True
    return mask, np.sort(cols)


def apply_mask(X, mask):
    out = X.copy()
    out[mask] = np.nan
    return out


def main():
    X, y = load()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'reports' / 'block_missing' / stamp
    out.mkdir(parents=True)
    print('输出目录:', out)

    rows, col_log = [], []
    for seed in SEEDS:
        tr_idx, te_idx = split_indices(y, seed)
        train_X, test_X, test_y = X[tr_idx], X[te_idx], y[te_idx]
        for rate in RATES:
            for layout in ['scattered', 'block']:
                rng_tr = np.random.default_rng([seed, 11])
                rng_te = np.random.default_rng([seed, 22])
                cols = None
                if layout == 'scattered':
                    m_tr = mask_scattered(*train_X.shape, rate, rng_tr)
                    m_te = mask_scattered(*test_X.shape, rate, rng_te)
                else:
                    m_tr, cols = mask_block(*train_X.shape, rate, rng_tr)
                    m_te, _ = mask_block(*test_X.shape, rate, rng_te)

                # 校验：两种布局的缺失总数必须完全一致
                assert m_tr.sum() == total_cells(*train_X.shape, rate)
                assert m_te.sum() == total_cells(*test_X.shape, rate)

                tr = apply_mask(train_X, m_tr)
                te = apply_mask(test_X, m_te)
                if cols is not None:
                    col_log.append({'seed': seed, 'missing_rate': rate,
                                    'n_selected_columns': int(len(cols)),
                                    'columns': ','.join(map(str, cols.tolist()))})

                for imp in IMPUTERS:
                    for det in DETECTORS:
                        scores, timing = fit_score(tr, te, imp, det, seed, DEFAULT)
                        row = {'seed': seed, 'missing_rate': rate, 'layout': layout,
                               'imputer': imp, 'detector': det,
                               'AP': average_precision_score(test_y, scores),
                               'ROC_AUC': roc_auc_score(test_y, scores),
                               'train_missing_actual': float(m_tr.mean()),
                               'test_missing_actual': float(m_te.mean()),
                               'total_seconds': timing['total_seconds']}
                        rows.append(row)
                        print(f"  seed={seed} rate={rate:.0%} {layout:9s} {imp:6s}+{det:7s} "
                              f"AP={row['AP']:.4f} AUC={row['ROC_AUC']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / 'results.csv', index=False, encoding='utf-8-sig')
    summary = df.groupby(['layout', 'missing_rate', 'imputer', 'detector']).agg(
        n=('AP', 'size'), AP_mean=('AP', 'mean'), AP_std=('AP', 'std'),
        ROC_AUC_mean=('ROC_AUC', 'mean'), ROC_AUC_std=('ROC_AUC', 'std'),
        seconds_mean=('total_seconds', 'mean')).reset_index()
    summary.to_csv(out / 'summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(col_log).to_csv(out / 'selected_columns.csv', index=False, encoding='utf-8-sig')

    # 同一条件下 集中 - 分散 的差值，这是本实验真正要看的数
    pivot = summary.pivot_table(index=['missing_rate', 'imputer', 'detector'],
                                columns='layout', values='AP_mean')
    pivot['AP_diff_block_minus_scattered'] = pivot['block'] - pivot['scattered']
    pivot.to_csv(out / 'layout_contrast.csv', encoding='utf-8-sig')

    meta = {'created_at': stamp, 'seeds': SEEDS, 'rates': RATES,
            'block_fraction': BLOCK_FRACTION, 'n_configs': len(df)}
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n完成，共', len(df), '个配置。')
    print(pivot.round(4).to_string())


if __name__ == '__main__':
    main()
