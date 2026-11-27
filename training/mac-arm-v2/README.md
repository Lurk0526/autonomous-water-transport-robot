# M2 Pro 机械臂实验包 v0.2

修复关节独立限速破坏姿态、按固定时钟切换抓放阶段、放回残差过大、奖励目标随动作移动、评估场景不固定及 TensorBoard 依赖冲突。继续使用用户 STEP 网格和 MuJoCo 原生 CPU。

**优先使用规则基线并标定夹爪运动；当前 MG23 仍是四指径向包络近似。** 用户确认实机能稳定夹起满载 650g 碗，本包尚未证明能准确复现实机接触，不把仿真失败解释为实机夹力不足。

单独解压到 `mac-arm-v2`，保留旧 v1 目录和训练记录：

```bash
bash setup.sh
bash run.sh inspect
bash run.sh baseline --episodes 20 --seed 30000 --output outputs/check20
bash run.sh diagnose --seed 30000 --output outputs/diagnose-30000
bash run.sh gui --experimental-model --seed 30000
```

阅读 [Mac调参与仿真指南.md](Mac调参与仿真指南.md) 和 [调整验证报告](validation/调整验证报告.md)。观测、控制和奖励已改变，旧 v1 策略不能加载或续训到 v2；v2 训练需从头开始。短训用于验证软件链路，不代表任务达标。实机连接、伺服指令导出、视觉接入和流体仿真尚未实现。
