# 测试证据

原有 [视觉台架及夹具照片](../assets/images/README.md) 与 P1–P5 报告保留。2026-11-02 从 Jetson 补入 [软件证据目录](runtime-import-20261102/)：

| 证据 | 范围 |
| --- | --- |
| `navigation-20261101/` | 原有 10 项循迹、38 项备用、4 项任务、27 组敏感性场景及碰撞反例的汇总/测试记录；全部为合成场景 |
| `bowl-preflight-previous/` | 原有监督器故障、数值和奖励审计；真实抓取、物理水与训练准入均未通过实物验收 |
| `road-model/` | 原有 4060 训练、独立测试和审计；本次没有重训或重新评估模型 |
| `import-verification.json`、`portable-tests.txt` | 本次接入后实际执行的软件检查，命令与结果均在文件内 |
| `source-manifest.json` | 新增工程/证据文件的大小和 SHA-256；不含密码、令牌、虚拟环境和权重 |

原始 rosbag、实时图像、训练数据和二进制权重未随本次上传；本机存储位置见 [数据](../data/README.md) 和 [模型](../models/README.md) 索引。导入时将证据文本中的本机项目路径替换为位置占位符，文件作为原有结果的整理副本；其中已有源码哈希与代码对应，整理后的证据文件哈希另见本次 manifest。

导航及任务的实际运行入口见 [simulation/](../simulation/README.md)。软件通过、台架相机输入和照片不能证明实车闭环、持续载水、30 km/h、振动改善或 Sim-to-Real 已完成。

新增实测按 [验证安排](../docs/verification.md) 记录日期、硬件、代码版本、环境/载荷、原始文件、失败现象与适用范围。
