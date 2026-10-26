# 缺失数据条件下表格异常检测的鲁棒性评估与轻量改进

第八组，成员刘子杨、纪景瑞、黎博文。研究方向拟定，创新尚待查新与验证。

## 当前可运行内容

`run_smoke.py`：一个种子、完整数据、中位数与Isolation Forest。  
`run_baseline.py`：真实Arrhythmia数据、两种填补与两种检测器、3个种子、三个缺失比例。无需命令行参数。  
`verify_protocol.py`：检查数据划分、掩码、评分与测试特征隔离。  
`make_report.py`：根据最近正式结果更新阶段报告。

PyCharm使用步骤与真实解释器路径见[打开项目指南](打开项目指南.md)。实验配置在`research/config.py`。依赖锁定在`requirements-research.txt`，不覆盖原框架依赖文件。

## 研究状态

已完成传统基础实验和指定文献、代码、数据核查。尚未完成ImAD方法复现、MacrOData基准运行、集中缺失对照和稳定性融合。基础实验不代表论文复现或创新成功。

## 资料

- [阶段报告](reports/presentation/stage_report.md)
- [研究协议](docs/research_protocol.md)
- [文献核查与综述起稿](docs/literature_review.md)
- [五周计划与分工](docs/plan_and_roles.md)
- [实际工作记录](docs/tonight_log.md)

原README、main.py、src与notebooks为仓库已有通用框架，完整保留。新研究模块在research，复用原有data、docs、reports目录约定。
