# 数据挖掘技术 — 课程小组作业

> 2026年10月25日研究进展：新增“缺失数据条件下表格异常检测”专用工程与阶段实验。请先阅读 [研究说明](README_RESEARCH.md) 和 [打开项目指南](打开项目指南.md)，点击运行 `run_baseline.py`。下方原始通用框架说明保留供参考。

> **课程**：数据挖掘技术（数字挖掘技术）
> **类型**：小组作业
> **状态**：🚧 进行中

围绕 **（待填写：研究问题 / 数据集主题，例如"基于 XXX 数据集的 YYY 预测与分群分析"）**
完整走通数据挖掘的标准流程：

**数据获取 → 数据理解(EDA) → 数据预处理 → 特征工程 → 建模 → 评估 → 结论与展示**

---

## 一、研究目标

| 编号 | 目标 | 对应交付物 | 状态 |
|---|---|---|---|
| G1 | 数据获取与描述性分析，说明数据质量与分布特征 | `notebooks/01_eda.ipynb`、`reports/figures/` | ⬜ |
| G2 | 完成缺失值/异常值/编码/标准化等预处理 | `src/preprocess.py` | ⬜ |
| G3 | 构造有业务含义的特征并筛选 | `src/features.py` | ⬜ |
| G4 | 建立并对比多个模型，选出最优方案 | `src/train.py` | ⬜ |
| G5 | 用多种指标评估并给出可解释的结论 | `src/evaluate.py`、`reports/` | ⬜ |

---

## 二、小组成员与分工

| 姓名 | 学号 | 主要负责 | 具体工作 |
|---|---|---|---|
| （待填写） | | 数据获取与 EDA | 数据收集、缺失值分析、可视化 |
| （待填写） | | 预处理与特征工程 | 清洗、编码、标准化、特征构造 |
| （待填写） | | 建模与调参 | 模型训练、交叉验证、超参数搜索 |
| （待填写） | | 评估与报告 | 指标对比、图表、答辩材料 |

详细的排期与会议记录见 [`docs/group-work.md`](docs/group-work.md)。

---

## 三、目录结构

```
data-mining-course/
├── README.md               # 本文件：项目说明
├── requirements.txt        # Python 依赖
├── main.py                 # 一键跑通完整流程（P0 入口）
├── data/
│   ├── raw/                # 原始数据（不要手工修改；大文件不入库）
│   └── processed/          # 预处理后的中间数据
├── notebooks/
│   └── 01_eda.ipynb        # 探索性数据分析
├── src/                    # 可复用的代码模块
│   ├── config.py           # 全局配置（路径、随机种子、目标列）
│   ├── data_loader.py      # 数据读取（含演示数据回退）
│   ├── preprocess.py       # 数据清洗与预处理
│   ├── features.py         # 特征工程与筛选
│   ├── train.py            # 模型训练与交叉验证
│   ├── evaluate.py         # 评估指标与报告
│   └── visualize.py        # 可视化图表
├── reports/
│   └── figures/            # 输出的图表、评估报告
└── docs/
    └── group-work.md       # 分工、排期、会议记录
```

---

## 四、环境准备

需要 **Python 3.10+**（建议 3.11）。

```bash
# 1. 克隆仓库
git clone <你的仓库地址>
cd data-mining-course

# 2. 创建并激活虚拟环境（Windows）
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
# python3 -m venv .venv && source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 五、快速开始

```bash
# 用内置演示数据（sklearn iris）跑通完整流程，确认环境没问题
python main.py --demo

# 用真实数据跑（把数据文件放到 data/raw/ 下，程序会自动识别 csv/excel）
python main.py --data data/raw/your_dataset.csv

# 指定模型
python main.py --data data/raw/your_dataset.csv --model random_forest
```

运行后会在 `reports/` 下生成评估报告与图表。

也可以直接用 notebook 做交互式分析：

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## 六、数据集说明

| 项目 | 内容 |
|---|---|
| 数据来源 | （待填写：如 Kaggle / 天池 / 学校提供 / 网络爬取） |
| 数据规模 | （待填写：样本数 × 特征数） |
| 目标变量 | （待填写） |
| 任务类型 | （待填写：分类 / 回归 / 聚类 / 关联规则） |
| 数据字典 | （待填写：各字段含义） |

> ⚠️ 原始数据如果超过 50MB，请放到网盘并在 README 里写下载方式，不要直接提交到 GitHub。

---

## 七、方法与模型

- **预处理**：缺失值填充（均值/中位数/众数/模型填充）、IQR 异常值处理、类别特征 One-Hot 编码、数值特征标准化
- **特征工程**：业务衍生特征、分箱、相关性/方差筛选
- **模型**：逻辑回归、决策树、随机森林、KNN、SVM（可用 `src/train.py` 一键对比）
- **评估**：准确率、精确率、召回率、F1、混淆矩阵、ROC-AUC；交叉验证

---

## 八、实验结果

| 模型 | 准确率 | 精确率 | 召回率 | F1 | 备注 |
|---|---|---|---|---|---|
| （待填写） | | | | | |

结论与讨论：（待填写）

---

## 九、常见问题

1. **matplotlib 图表中文变成方框** —— 在 `src/visualize.py` 中已尝试自动设置中文字体（SimHei / Microsoft YaHei），如仍异常请检查系统是否安装该字体。
2. **`.venv\Scripts\activate` 报错** —— 用 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` 临时放开后重试。
3. **数据文件找不到** —— 确认文件放在 `data/raw/` 下，或显式传 `--data` 参数。

---

## 十、参考与致谢

- 课程教材/课件：（待填写）
- 参考资料：（待填写）
