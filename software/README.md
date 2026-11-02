# 上位机与视觉软件

2026-11-02 接入 Jetson 上已有工程，完整保留在 [jetson/](jetson/README.md)，不拆散其 Python 导入、配置和脚本路径。原目录结构中的 `cart/`、`vision/`、`sim/`、`tools/` 均从 `software/jetson/` 作为工作目录运行。

| 模块 | 入口与说明 | 当前范围 |
| --- | --- | --- |
| 三色循迹主导航、无标线备用 | [cart/](jetson/cart/README.md)、`run_cart_sim.sh` | 合成图像闭环、路径跟踪、绕障及失效处理；尚未接真实底盘 |
| Gemini MAX 道路 RGB-D 感知 | [道路感知说明](jetson/docs/road-perception-preinstall.md)、`open_road_perception.sh` | ROS 2 Jazzy 台架感知、深度关联、录制回放；装车外参未测 |
| OpenMV 碗识别与机械臂跟随 | [跟随说明](jetson/docs/bowl-follow.md)、`start_bowl_vision.sh` | 图像跟随与串口反馈；不等于自动三维抓取 |
| 训练、标注、评估和导出 | [碗训练](jetson/docs/target-bowl-training.md)、[道路训练](jetson/docs/road-training-plan.md)、[4060 指南](jetson/docs/RTX4060训练指南.md) | 已有脚本与配置；完整数据和权重另行保存 |
| 真实输入检查与采集 | `cart_preflight.sh`、`cart_capture.sh` | 只订阅/录制，不发布底盘运动指令 |

先读 [本次接入记录](../docs/runtime-import-20261102.md) 和 [运行入口](jetson/START_HERE.md)。原有 [环境盘点脚本](../tools/check_jetson_env.py) 保留，运行工程也有其自身的环境验证工具。

本次未收到团队定制 STM32 底盘工程、独立部署的 OpenMV 板上程序或完整自定义 ROS 包，不创建占位控制实现。厂家原始资料不等于团队实车固件。
