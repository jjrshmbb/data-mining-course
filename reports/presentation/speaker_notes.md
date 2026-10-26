## 第1页 选题

我们选择缺失条件下的表格异常检测可靠性问题。本轮刚完成基础工程和传统流程实验，没有声称此前三周已经完成研究。项目仍处于起步阶段。

## 第2页 研究问题与当前进度

三个问题按依赖顺序推进。今晚只验证了第一个问题的基础实验，第二和第三个问题尚未实验。全部实际推进由管理员与AI助手完成，成员分工是后续安排。

## 第3页 近期研究与创新边界

来源：MacrOData https://arxiv.org/html/2602.09329v1 ，KDD官网 https://kdd2026.kdd.org/papers/ ，官方DOI https://doi.org/10.1145/3770855.3817520 。ImAD https://proceedings.neurips.cc/paper_files/paper/2024/file/f99f7b22ad47fa6ce151730cf8d17911-Paper-Conference.pdf 。M-DEW https://arxiv.org/html/2405.00182v1 。本轮只核查指定章节和代码，并未完成全面综述。SEAOP 2024摘要与可见片段也涉及扰动稳定性筛选模型，仍待补读全文。近期论文任务拟选ImAD，需向老师确认是否强制从KDD2026入口选论文。

## 第4页 真实数据与划分

数据来自ImAD作者仓库a4963fc976fbb10759717dd6afea51320a5d2639中的data/Arrhythmia/complete_data.npy。文件SHA256为0ccb314f6a395c5e38d06694ae7fad9f4b7d10ad162cb03d47eb047a26ff1810。代码规定0正常1异常。按split=1数量规则新生成并保存划分，不称为论文原索引。论文表1写正常320异常132，与文件386/66不一致，可能混用划分数但尚未证实。HF接口超时，故本轮没有使用MacrOData基准数据。

## 第5页 基础实验协议

配置来源：research/config.py与正式运行config.json。种子42、2026、3407同时控制划分、MCAR掩码与Isolation Forest。缺失为伯努利采样，实际比例记录于results.csv。所有方法复用同一掩码。StandardScaler只在含缺失训练集拟合，随后在缩放空间拟合中位数/KNN填补器，最后训练检测器。LOF novelty=True仅评分新测试样本。IF200棵树，KNN5邻居，LOF20邻居。AP使用average_precision_score，不称为梯形PR-AUC。计时在一次预热后分别记录预处理、拟合、评分之和。

## 第6页 AP的变化并非统一下降

来源：C:\Users\HP\Documents\Codex\2026-10-25\new-chat\outputs\DataMiningResearch\reports\runs\20261025_202939_baseline_07f5d4/summary.csv。绘制3个种子的均值，完整均值±样本SD见CSV、报告及metrics.png。AP为average_precision_score。测试阳性比例0.5，仅是参考，不等于当前小样本随机排名AP的精确期望。该图展示传统对照，无创新方法。

## 第7页 ROC-AUC与计算成本

来源：C:\Users\HP\Documents\Codex\2026-10-25\new-chat\outputs\DataMiningResearch\reports\runs\20261025_202939_baseline_07f5d4/summary.csv。左图为3种子均值，标准差见附表。右侧为30%MCAR下预处理加拟合加评分的平均秒数，已预热。中位数IF的AP略升而ROC-AUC略降，不能用一个指标宣布改善。KNN填补带来额外成本，当前只有小数据单机时间，不能作为大规模复杂度证明。

## 第8页 候选假设与证据边界

稳定性尚未实验。候选方案先用无标签校准集额外遮蔽已观测单元，记录Spearman排序相关、固定top-k重叠、分数IQR，识别常量或大量并列的塌缩。与检测准确率的关系只作预登记后的事后诊断，不用测试标签反向确定权重。若进一步融合，统一用校准分数映射，比较等权与稳定性加权，并计算额外成本。若不优于等权或稳定性反向，就保留负结果。

## 第9页 后续五周计划与拟分工

计划详见docs/plan_and_roles.md。第1周10月26日至11月1日，第2周11月2日至8日，第3周11月9日至15日，第4周11月16日至22日，第5周11月23日至29日，第6周11月30日至12月6日仅缓冲。最终截止未确定。各成员状态待本人确认与参与，今晚实际由管理员推进，不能说成三人均已完成。汇报可请求老师确认近期论文是否必须来自KDD2026，以及当前复现方法与基准结合是否符合要求。