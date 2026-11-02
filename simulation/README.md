# 导航仿真与抓放流程审计

2026-11-02 接入可运行的本机仿真。实现保留在完整工程内，避免复制后产生两套代码：

| 内容 | 实现 | 运行 |
| --- | --- | --- |
| 循迹主导航、无标线备用 | [line_free_cart](../software/jetson/sim/line_free_cart/RUN.md) | 在 `software/jetson/` 执行 `python3 -m sim.line_free_cart.run` |
| 标准库备用导航 | 同上 | 追加 `--backup-only`，不需要 NumPy/OpenCV |
| 抓放任务监督器、数值与奖励审计 | [bowl_transport](../software/jetson/sim/bowl_transport/RUN.md) | 在 `software/jetson/` 执行 `python3 sim/bowl_transport/preflight.py` |
| Isaac 起始 USD 场景与云端准备 | 同一 `bowl_transport/` 目录 | 版本、操作与未验证范围见其 RUN.md |

导航套件包含 10 项三色循迹、38 项无标线备用、4 项任务流程、27 组噪声/延迟组合和碰撞反例。参数与观测是合成数据；停车/拒绝出发按预期行为统计，不能计作完成运输。

抓放模块的夹持、承托和松脱信号是模拟反馈；没有真实机器人关节资产、接触动力学、物理水或训练完成的抓碗策略。Isaac Lab 训练准入仍为 false。本次没有启动云端训练。

本机已有第三版命名的原生 CAD 保留在 [工程资产](../software/jetson/assets/cad/chassis-v3/manifest.json)，与 [第二版机械资料](../hardware/cad/v2/README.md) 分开保存。目录命名及原始校验清单不能证明它是当前 650 mm 实车的最终版本，也不能替代 URDF/articulation 和装配依赖检查。

已有软件证据见 [results/](../results/README.md)，完整任务边界见 [接入记录](../docs/runtime-import-20261102.md)。
