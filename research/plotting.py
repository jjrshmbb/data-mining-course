import os
from .config import ROOT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

COLORS = ['#125d98', '#008a76', '#cd5c24', '#8c4d9d']

def plots(df, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for (imp, det), color in zip(df.groupby(['imputer','detector']).groups, COLORS):
        sub = df[(df.imputer == imp) & (df.detector == det)]
        g = sub.groupby('missing_rate')
        for ax, metric in zip(axes, ['AP','ROC_AUC']):
            stats = g[metric].agg(['mean','std']).fillna(0)
            x = stats.index.to_numpy()*100
            ax.errorbar(x, stats['mean'], yerr=stats['std'], marker='o', capsize=4,
                        label=f'{imp} + {det}', color=color, linewidth=1.8)
            ax.set_xlabel('MCAR missing entries (%)')
            ax.set_ylabel('Average Precision (AP)' if metric == 'AP' else 'ROC-AUC')
            ax.set_xticks([r * 100 for r in sorted(df.missing_rate.unique())]);
            ax.set_ylim(0.35, 1.01)

            ax.grid(alpha=.2)
    axes[0].axhline(.5, color='#777777', linestyle='--', linewidth=1, label='test prevalence = 0.50')
    axes[1].axhline(.5, color='#777777', linestyle='--', linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('Arrhythmia: normal-only training, both train/test masked')
    fig.text(.5,.115,f'Mean ± sample SD, {df.seed.nunique()} seeds. Variation includes split, mask and model.',
             ha='center',fontsize=9,color='#555555')
    fig.tight_layout(rect=[0,.13,1,.94])
    fig.savefig(out/'figures/metrics.png', dpi=180)
    plt.close(fig)
    top = df.missing_rate.max()
    g = df[df.missing_rate == top].groupby(['imputer', 'detector']).total_seconds.agg(['mean', 'std']).fillna(0)

    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar([f'{a}\n+ {b}' for a,b in g.index],g['mean'],yerr=g['std'],capsize=4,color=COLORS)
    ax.set_ylabel('Seconds (preprocess + fit + score)')
    ax.set_title(f'Runtime at {top:.0%} MCAR, single process')

    ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(out/'figures/runtime.png',dpi=180); plt.close(fig)

def analysis(df, paired, out):
    lines = ['# 阶段实验分析', '',
             '本轮是传统基础流程实验。尚未运行 ImAD 方法，尚未实现候选创新。', '',
             f'实际成功运行 {len(df)} 个流程配置，包含 {df.seed.nunique()} 个种子。完整数据、10% 和30% 为预设条件，未依据测试成绩选择数据或参数。', '',
             '## 实际结果', '', '|填补|检测|缺失率|AP 均值 ± SD|ROC-AUC 均值 ± SD|秒均值|',
             '|---|---|---|---|---|---|']
    for keys,s in df.groupby(['imputer','detector','missing_rate']):
        imp,det,rate=keys
        sd=lambda x: 0.0 if len(x)<2 else x.std(ddof=1)
        lines.append(f'|{imp}|{det}|{rate:.0%}|{s.AP.mean():.4f} ± {sd(s.AP):.4f}|{s.ROC_AUC.mean():.4f} ± {sd(s.ROC_AUC):.4f}|{s.total_seconds.mean():.3f}|')
    lines += ['', '## 相对完整数据的配对变化', '']
    if (paired.missing_rate == .3).any():
        for (imp,det),s in paired[paired.missing_rate == .3].groupby(['imputer','detector']):
            lines.append(f'- {imp} + {det}：30% MCAR 的 AP 平均变化 {s.delta_AP_vs_complete.mean():+.4f}，ROC-AUC 平均变化 {s.delta_ROC_AUC_vs_complete.mean():+.4f}。')
    lines += ['', '## 证据边界', '',
      '只使用一个小型高维数据集，且测试异常比例为50%，AP不可直接外推到低异常率应用。0.50虚线仅为测试阳性比例参考。',
      '少量种子同时改变划分、缺失掩码和森林随机性，标准差不是置信区间，也不是跨数据集统计显著性。缺失率使用伯努利采样，实际比例见CSV。',
      '曲线可以非单调。单个条件下指标改善可能来自随机性或填补的平滑作用，不能据此宣布缺失有益。',
      '这里只验证 MCAR，同时遮蔽训练与测试特征。尚未验证集中缺失、MAR/MNAR、仅测试时缺失或稳定性融合。',
      '运行时间来自当前机器单进程测量，包含填补、缩放、拟合和评分，不包含下载、绘图和指标计算。', '',
      '## 下一步', '',
      '先完成 ImAD 官方方法的独立环境迁移与单种子检查，再与其按相同划分和掩码比较。补充第二个预先登记的数据集，扩至5个种子。',
      '集中缺失实验需匹配总缺失量、随机选列并记录列索引，不能根据标签或性能选择特征。稳定性先做诊断，再决定是否值得融合。']
    (out/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
