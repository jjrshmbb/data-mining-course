# 本轮实际工作记录

日期：2026年10月25日。执行者：管理员与AI开发助手。未记录为另外两名成员已开展的工作，不补造历史进度。

- 检查实际Python路径与版本。发现此前Documents/DataMiningResearch只有.venv，无项目代码或Git仓库。
- 在当前工作目录outputs/DataMiningResearch下载第八组仓库，基于35c76db810ec5d382c64890ea84ece779fa66cce建立独立研究分支。保留原README、main.py与src通用框架。
- 首次Git连接遇到Windows schannel证书凭据错误，使用本次命令级OpenSSL后端成功。未修改全局Git设置。
- 获取ImAD与MacrOData作者仓库、论文正文和KDD官方条目。记录仓库提交和下载失败情况。
- HF本机接口超时；MacrOData仓库未提供可用下载函数，未取得该基准实验文件。转用ImAD作者随代码提供的Arrhythmia数据。
- 核查npy真实形状、标签与缺失值。保存原始文件散列。发现论文表1与文件类别数存在差异，予以记录。
- 初次联网安装在受限缓存路径阻塞，另有部分下载缓慢。保留安装日志。改用本机已能正常导入的科学计算包复制到项目独立虚拟环境，保留dist-info与DLL，验证不引入系统site-packages，并记录实际版本。这不是原论文环境。
- 编写专用研究入口、配置、数据验证、训练测试分离、MCAR模拟、四种传统流程、AP与ROC-AUC、计时和结果保存代码。
- 首次单流程检查在系统Python环境真实运行完成，单独保留smoke目录。完整正式轮次使用随后验证的独立环境，具体次数、指标与时间以reports/runs为准。

未完成项：ImAD方法运行、MacrOData基准实验、第二个数据集、集中缺失对照、稳定性诊断与融合、全面综述。
