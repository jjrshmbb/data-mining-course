# 文献核查与综述起稿

核查日期：2026年10月25日。以下“正文核查”指本轮对指定章节与代码的核对，不代表三名成员已经通读论文、掌握证明或完成复现。尚无完整方法复现结果。

## 已核实的核心文献

|文献|类型与版本|本轮实际核查范围|与项目关系|实现成本与当前状态|
|---|---|---|---|---|
|MacrOData: New Benchmarks of Thousands of Datasets for Tabular Outlier Detection|KDD 2026 Datasets and Benchmarks；arXiv 2602.09329v1|正文第3节数据协议及部分评估说明、KDD官网内嵌论文条目、作者GitHub数据加载代码、HF组织与文件页|提供新基准与正常样本训练划分，区分语义异常OddBench、类别转换OvRBench和合成SynBench|小数据子集上传统流程成本低。作者仓库下载函数仍是TODO，路径指向作者服务器。HF本机接口超时，未取得基准数据，未运行该基准|
|Unsupervised Anomaly Detection in The Presence of Missing Values，ImAD|NeurIPS 2024 正式方法论文；会议版PDF 33页|正文引言、方法3.3与实验设置部分、附录J数据及参数、官方README、data/tabular.py、requirements.txt、main.py、训练器和网络入口|直接研究缺失下异常检测与填补偏差，是本项目计划复现的近期方法|中等成本，涉及三个网络与Sinkhorn损失。仓库依赖大量Linux本地路径，不能在Windows原样安装。已取到官方附带数据，方法尚未运行|
|M-DEW: Extending Dynamic Ensemble Weighting to Handle Missing Values|arXiv 2405.00182v1，2024-04-30；当前仅按已核实预印本记录|正文第3节算法1和2、第4节实验设置、摘要与版本页|已提出填补与预测流程的动态加权。其阶段2利用标签误差估计局部能力，与无标签异常检测的权重来源不同|不能将其直接当作无标签异常检测方法。未核实可运行的官方代码，未复现|
|SEAOP: a statistical ensemble approach for outlier detection in quantitative proteomics data|Briefings in Bioinformatics 25(3), 2024, bbae129，正式期刊|只核查出版信息、摘要和检索可见方法片段，正文访问受限|已有基于扰动与稳定性筛选异常检测模型的相关思想，提示“稳定性选择”并非当然新颖|作为补读对象，不能据摘要断言与本项目完全相同；尚未核查代码或实验|

## 来源

1. MacrOData正文：https://arxiv.org/html/2602.09329v1 。KDD官方入口：https://kdd2026.kdd.org/papers/ 。官方条目DOI：https://doi.org/10.1145/3770855.3817520 。代码：https://github.com/xyvivian/macrOData ，核查提交 `10d486eed1b073f052c29554e8a168510692325c` 。数据：https://huggingface.co/MacrOData-CMU 。
2. ImAD会议页：https://proceedings.neurips.cc/paper_files/paper/2024/hash/f99f7b22ad47fa6ce151730cf8d17911-Abstract-Conference.html 。正文：https://proceedings.neurips.cc/paper_files/paper/2024/file/f99f7b22ad47fa6ce151730cf8d17911-Paper-Conference.pdf 。代码：https://github.com/jicongfan/ImAD-Anomaly-Detection-With-Missing-Data ，核查提交 `a4963fc976fbb10759717dd6afea51320a5d2639` 。
3. M-DEW正文：https://arxiv.org/html/2405.00182v1 。版本页：https://arxiv.org/abs/2405.00182 。
4. SEAOP出版页：https://academic.oup.com/bib/article/25/3/bbae129/7638267 。摘要：https://pubmed.ncbi.nlm.nih.gov/38557674/ 。

## 候选改进的重叠与调整

“先填补再异常检测”和随缺失率作比较已被ImAD研究，不能把这个实验框架当作原创方法。“多个填补与模型流程动态加权”与M-DEW的核心框架重叠。扰动稳定性参与异常模型筛选也有SEAOP等相关工作，因此暂不使用“首次提出”或“创新算法”表述。

本项目可保留的待验证差别是：以无标签校准特征上的缺失扰动为依据，在分散与集中两类缺失布局下，检验排序稳定性是否能反映检测质量，并显式识别分数塌缩造成的虚假稳定。这个差别是研究假设，尚不能证明文献空白。需要继续按下述检索范围查新。

最小调整为先做“稳定性诊断的失效条件研究”，使用排序相关和top-k重叠，加入常量分数/并列分数保护。若诊断有价值，再比较等权融合与预先冻结规则的稳定性加权。若无效，保留负结果与边界，完成可靠性评估。不能把微小保护规则自动称为论文级创新。

## 综述组织框架

一类工作将缺失处理作为预处理，以统计或邻居填补连接传统异常检测器。另一类工作联合建模缺失恢复与异常识别，ImAD属于这一方向。第三类工作关注模型选择与集成，但必须区分有标签验证误差、无标签内部指标、历史元学习等不同信息来源。基准工作解决数据覆盖和评估协议，不替代方法研究。

现有证据提示，应把填补质量、检测质量与分数稳定性分别评价。填补更接近正常训练分布，不保证更能暴露异常。排序稳定也可能意味着模型对信息损失不敏感，甚至所有样本得到接近的分数。我们将用受控缺失实验检验这些关系，而不将它们预先设为正相关。

## 后续补读与查新登记

- 时间范围先覆盖2023年至2026年10月25日，优先正式会议与期刊，预印本单列。检索词：missing tabular anomaly detection、missingness-aware ensemble、unsupervised outlier model selection、perturbation rank stability、imputation bias。
- 建议两轮筛选：题目/摘要保留与表格缺失或无标签模型评价直接相关的研究，全文再记录训练协议、缺失机制、权重是否用标签、数据版本、代码状态及局限。
- 第一周补到8至12篇相关论文，逐篇标记“摘要”“指定章节”“全文”，分别保存读者、日期和问题清单。当前只整理了3篇核心正文核查记录和1篇补读条目，不能称为完成子领域全面综述。
- 无标签模型选择可补查MetaOD、DSV及内部评价综述；未核实正文前仅作为检索线索，不写其算法结论。

## 必须保留的复现差异

ImAD论文表1列Arrhythmia正常320、异常132。附带npy实际标签为正常386、异常66，split=1代码会产生训练320与测试132。当前不能确定论文表格是否混用了划分和类别统计，不能擅自“修正”论文。今晚依据实际文件和代码规则运行，并明确AP与论文AUPRC可能计算方式不同，暂不作直接数值复现比较。
