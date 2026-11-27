# Mac v0.2 检查回传说明与验收报告

生成时间：2026-11-17T21:14:20.834946+08:00（北京时间）

**v2 软件验收与 1024 步短训已完成；200 场景 Mac 零残差基线为 179/200（89.5%），尚未达到 95% 目标。** 本轮按新版交接范围执行，没有启动 10 万或 30 万步正式训练。

## 1. 设备、目录与实际执行范围

- 同一台 Apple M2 Pro / 16 GiB Mac；系统为 macOS-27.3-arm64-arm-64bit，原生 arm64，Python 3.12.16。
- 项目目录：training/mac-arm-v2
- 本轮实验目录：training/mac-arm-v2/outputs/mac-v2-check-20261117-210301
- 检查日志：training/mac-arm-v2-handoff-20261117/Mac检查日志/20261117-210301
- 依赖：MuJoCo 3.3.7、PyTorch 2.8.0、SB3 2.7.0、Gymnasium 1.2.1、TensorBoard 2.19.0、setuptools 80.9.0。
- 外层交接包 11 项 SHA256 全部匹配；安装脚本内层校验通过，完全使用离线依赖。v1 目录、权重、报告及原始交接包继续保留。

## 2. 验收结果

| 项目 | Mac 实测结果 |
| --- | --- |
| 离线安装、pip check、模型 inspect | 全部退出码 0 |
| 测试 | 21/21 通过，测试日志耗时约 10.25 s |
| 原生 GUI | 独立运行 CAD 模式 45 s，正常显示机械臂、四指夹爪、碗和台面，正常退出；不是实物联动校验 |
| 开发基线 30000—30099 | 93/100，93.0%；运行 60.9 s |
| 复核基线 40000—40199 | 179/200，89.5%；运行 118.1 s |
| 不可达场景 | 开发组 0/100，复核组 0/200；未剔除任何失败回合 |
| PPO smoke | 种子 42，2 个 spawn 环境，实际 1024 步；权重、检查点、CSV、TensorBoard 均已生成 |
| 运行中源码完整性 | unchanged_during_run=true |
| 最终盲测 50000—50199 | 未使用，继续保留 |

两组完整 JSON 均保留全部 trials。复核组 Wilson 95% 区间为 84.5%—93.0%；未达到 95% 点估计目标。

开发组失败：unexpected_collision：4；joint_tracking_limit：3。

复核组失败：unexpected_collision：5；support_missing_after_lower：1；joint_tracking_limit：12；dropped_bowl：1；lift_hold_not_verified：2。

## 3. 与随包 Linux 参考的区别

同一批 40000—40199，随包 Linux 参考为 186/200（93.0%），本机 Mac 为 179/200（89.5%），差 -3.5 个百分点。源码、配置、候选模型的三个 SHA256 均一致；差异的具体来源尚未定位，不能将其归因于某个已确认的物理参数。

v2 的碰撞与阶段稳定检查比 v1 更严格，且此处种子与旧 Mac 61% 的种子不同，因此不把两者百分比相减写成同口径算法提升。本轮未在 Mac 重新运行 v1 与协调限速消融。

## 4. 成功、失败诊断与接触证据

| 种子 | 结果 | 终止阶段 | 原始证据 |
| --- | --- | --- | --- |
| 30003 | 成功 | settle_place | diagnose-30003/episode.json、trace.json、replay.json |
| 40004 | unexpected_collision | lower | diagnose-40004/episode.json、trace.json、replay.json |
| 40010 | support_missing_after_lower | lower | diagnose-40010/episode.json、trace.json、replay.json |
| 40022 | joint_tracking_limit | close | diagnose-40022/episode.json、trace.json、replay.json |

成功种子 30003 完成抓放，记录 hold_verified=true、最终连续稳定 0.6 s，总模拟时间约 11.7 s。

复核组首次碰撞记录：platform / instance_148_hard：4 次；platform / instance_161_hard：1 次。这些几何体是硬指节代理与台面接触。首次碰撞按 1 ms 物理子步保留，详细阶段、时间、力和位置在原始 JSON 中；仿真力未经实机标定，不作为真实冲击力解释。

完整 trace 中保留泡沫与硬指节接触区别、关节目标与实际值、残差及阶段确认。CAD 窗口已观察到模型显示；本次没有实物全开、中间、夹稳状态的对照资料，不能判定真实 MG23 联动和泡沫接触已匹配。

## 5. 短训与曲线

短训固定评估仅使用 30000—30002 三个场景，基线为 3/3，最佳短训策略为 3/3，配对差为 0。该结果只验证软件流程，不能宣称学到了更可靠的抓碗策略。selection.json 仍推荐 baseline，hardware_ready=false。

TensorBoard 服务在短训前已启动，读取日志间隔为 1 s，地址为 http://127.0.0.1:6007/。内置浏览器页面后续失效，已改在 Safari 确认曲线显示到 1024 步。短训日志报告 time_elapsed 约 6 s，这不是安装与全部验收的总耗时；没有为了展示更长曲线而额外启动正式训练。仅凭最后确认的页面，不声称已经连续观察完整的短训动画过程。

本次使用 v2 原生 TensorBoard CLI，不使用 v1 临时 shim。pkg_resources 废弃提示仍存在，但 CLI 与曲线服务均成功启动。

图表保留在本轮实验目录 report/v2-results.png，HTML 报告已内嵌图表。

## 6. 相对原始 v2 的修改与未完成项

- 模型、配置、arm_training 源码、测试和依赖锁均未修改；没有加载或续训 v1 权重，没有改变摩擦、夹力、碰撞阈值、10 mm 落点容差、30 mm 离台或 2 s 保持标准。
- 首轮检查脚本未添加 --gui，原生桌面检查单独执行 45 s CAD 显示，并额外保留 gui-native.log、gui-native-status.json。其他检查步骤按交接脚本执行，11 个脚本步骤退出码均为 0。
- 原交接包从桌面复制到工作区，日志与回传文件写入工作区副本，原桌面交接包未修改。
- 实机动作的舵机指令、时间、反馈、三种开口状态、泡沫压缩量本次均未提供，仍未知。交接说明记载的“实机已夹起 650 g 并保持至少 2 s”是外部交接信息，本轮未独立复核该实机动作。
- 未完成真实联动标定、真实视觉闭环、液面/失水量仿真和实机部署；这些不属于本轮软件短训已通过的范围。

## 7. 回传内容与下一步

回传 ZIP 包括本说明、两组完整评估、四个诊断目录、模型检查、短训全部权重/检查点/事件/CSV/评估、实际源码与配置，以及完整检查日志。排除虚拟环境、运行时、wheel 和未修改的整套 CAD 网格。打包工具重新检查 ZIP CRC、文件哈希并生成 ZIP SHA256；文件仅在本地生成，未发送到外部。

优先检查闭合阶段的关节跟踪与硬指节碰台，结合实机成功动作校准开口和接触位置，再决定是否需要正式 PPO。50000—50199 继续留作未来选定方案后的最终盲测。
