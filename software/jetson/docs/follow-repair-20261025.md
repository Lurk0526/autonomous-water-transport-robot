# 2026-10-25 跟随再次无法启动

本文件记录用户确认全范围之前的排查。随后用户要求全部关节范围可启动并确认安全，当前配置及后续验证见 [全范围启动](full-range-follow-20261025.md)。

本次先对比了 `releases/bowl-follow-v1.0.0/`：修复前 `vision/`、`tools/`、`config/` 与验收版内容相同（忽略 Python 缓存）。
没有把其他试验参数覆盖到初版，也未修改发布快照。

## 实际故障

界面日志记录了先后发生的两类启动失败：未识别到清晰目标，以及
`Invalid calibrated bounds/direction for joint 2`。跟随服务未运行，旧跟随状态文件来自 10 月 23 日，不能代表本次连接。

独立连接检查确认 CH340 复位保护为 Y，启动广播为空、握手 AAA，六路实测反馈均正常：
000=1547、001=1366、002=1229、003=590、004=1495、005=1533。
初版的肘部范围为 P1437～2447，腕部为 P650～1800；当前两个关节均超出范围，足以解释位置校验失败。
这些是已经实测的运行范围，不是对舵机硬件全行程的断言。

同时发现相机、检测和 Isaac 节点仍尝试向旧地址 10.81.223.9 发送 DDS 数据，持续报告
`ddsi_udp_conn_write ... failed with retcode -1`，而当前 Wi-Fi 地址已是 172.20.10.2。
检测状态为 stale，保存的图像时间戳与当前时间严重不符。视觉进程存在并不意味着新图像已传递到识别节点。

## 上次实际修过什么

10 月 23 日最终 v1.0.0 修复了局部标定窗口误当行程、肘腕分配及平滑续接、
肘部停止后的负载回弹误判，并将参数及三个实测姿态模型保存到 config。
启动从六路当前位置建立会话，无需恢复固定姿态。
这一能力始终受已经验证的行程约束，不能解释成协议 P500～2500 内任意位置均已验证可用。
历史细节见 `smooth-current-pose-follow-20261023.md` 和 `arm-control-repair-20261023.md`。

## 本轮修改

- `config/cyclonedds-local.xml` 固定本机回环接口、单播发现和 localhost peer；
  `config/ros-transport.sh` 供项目 Python 和使用系统 Python 的 Isaac 启动器共用。
  显式设置的 DDS 配置或其他中间件保持优先。跨机接入需提供对应配置。
- 从当前位置启动前先检查实测行程，汇总所有越界关节及允许范围，避免无目标时掩盖位置问题。
  已选择姿态模型时也不能绕过该检查；失败保留原会话配置并关闭串口。
- 界面和 README 明确“已验证行程内无需固定初始姿态”。

配置依据：[Cyclone DDS 运行配置](https://cyclonedds.io/docs/cyclonedds/latest/config/runtime_config.html)、
[配置项参考](https://cyclonedds.io/docs/cyclonedds/latest/config/config_file_reference.html)。

## 验证和剩余条件

187 项测试通过；提取公共通信脚本后，3 项环境激活测试再次通过，Shell 语法检查通过。
独立两个 ROS 进程完成收发，没有通信错误。
重启视觉节点后，5 秒只读订阅收到 97 帧 RGB、92 条识别结果、93 帧 Isaac BGR，识别约 17.7 FPS。
相机与检测新进程日志未出现旧地址发送错误。

实机运行 `prepare_follow_trial.py --reuse-profile`，3.11 秒内明确报告
002 肘部 P1229、003 腕部 P590 超出允许范围；原配置 SHA256 不变。
本次只做连接、位置和视觉检查，没有发送运动目标，也未重新标定或扩大行程。
跟随保持停止，运动效果尚未恢复验收；需将关节调整到已验证范围内，或另行完成新范围的实测标定。

证据及修改前副本位于 `work/follow-repair-20261025/`：
`before-gui-errors.log`、`connection-status.json`、`local-transport-check.json`、
`live-startup-preflight.json`、`final-vision-check.json`、`tests.log`、`before/`。
