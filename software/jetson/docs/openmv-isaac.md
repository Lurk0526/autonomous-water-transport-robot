# OpenMV GUI 与 Isaac 接入

## 启动

```bash
cd "<项目目录>"
./start_vision_gui.sh
./start_isaac_vision.sh
```

第一个命令打开本机 Qt GUI，第二个命令启动 Isaac ROS 的 GPU 图像格式转换节点。GUI 里的暂停按钮会暂停取流；重新连接摄像头后可点击“继续取流”。

## 摄像头连接

机械臂自带的 OpenMV 固定在末端扩展板上。原厂现有总线线缆负责供电和串口控制，STM32 的 CH340 USB 端口不是相机视频接口。需要通过 **OpenMV 板自身的 Micro-USB 数据接口** 连接 Jetson。已实际连接成功：H7 Plus（OMV4P H7，32 MB SDRAM），固件 4.5.9，USB VID:PID `1209:abd1`，端口 `/dev/ttyACM0`。

依据本地手册 `1.14、机械臂J1视觉OpenMV拓展玩法` 第 9 页步骤，进行 USB 编程前先断开 OpenMV 扩展板上的总线连接。本程序无需同时控制舵机；可按该步骤让 OpenMV 单独通过 USB 取流。

GUI 仅自动探测匹配 OpenMV 的 USB 设备，拒绝向机械臂 CH340 发送相机调试命令。连接成功后暂停 OpenMV 原运行脚本，在 RAM 中运行仅含 `sensor.snapshot()` 的临时采图脚本；不保存到 `main.py`、不擦写固件、不发送舵机控制。

暂停或正常退出会停止临时脚本。原有闪存脚本仍保留，但不会自动重启原来的机械臂控制逻辑；以后重新上电时是否恢复原程序由板上原 `main.py` 决定。勿同时使用 OpenMV IDE 占用同一个 USB 串口。

板上原 `main.py` 是厂家“颜色识别做动作”程序，备份在 `work/openmv/main-before-streaming.py`，未覆盖原文件。取流客户端使用独立的命令/载荷 USB 阶段，并切换 CDC 波特率模式清理异常会话；4.5.9 使用 `GET_STATE` 同时读取帧状态和控制台信息。

## 图像数据流

```text
OpenMV Micro-USB 调试帧缓冲区
    → Jetson GUI（320×240 RGB）
    → /openmv/image_raw（sensor_msgs/Image, rgb8）
    → Isaac ROS ImageFormatConverterNode（GPU）
    → /isaac/openmv/image_bgr（sensor_msgs/Image, bgr8）
```

- `/openmv/camera_info`：当前未标定，K 保持全零，不伪造内参。
- `/openmv/status`：连接状态、实际发布帧数、是否标定和深度可用性。
- 时间戳暂为 Jetson 接收帧时刻，不宣称硬件同步或曝光时刻。
- 当前按单目 RGB 相机接入，不提供深度，不从二维像素直接驱动抓取。
- GUI 没有机械臂动作控制入口，也不打开机械臂控制板串口。

## 版本选择

本机为 Jetson Orin Nano Super、L4T 39.2.1、Ubuntu 24.04、ROS 2 Jazzy。使用固定的 **Isaac ROS 4.6** 官方 `noble-jetpack` 软件源，与现有 Jazzy 保持一致。Isaac ROS 5.0 已迁移 ROS 2 Lyrical，不混入当前工作环境。

官方源配置：`/etc/apt/sources.list.d/nvidia-isaac-ros.list`。安装计划、安装日志和版本记录位于 `work/isaac/`。本次选取图像处理与 NITROS 组件，不安装完整 Manipulation 模型或替换 CUDA/PyTorch。

Isaac Sim 的本地运行硬件要求与 Isaac ROS 不同。官方当前 aarch64 构建只支持 DGX Spark；本机不作为 Isaac Sim 渲染工作站。后续可在符合要求的 RTX 工作站运行 Isaac Sim，使用 ROS 2 Bridge 连接 Jetson，机器人模型、工具坐标及关节约束还需另行建立与验证。

参考：

- [Isaac ROS 4.6 系统要求](https://nvidia-isaac-ros.github.io/v/release-4.6/getting_started/index.html)
- [Isaac ROS 图像处理接口](https://nvidia-isaac-ros.github.io/v/release-4.6/repositories_and_packages/isaac_ros_image_pipeline/isaac_ros_image_proc/index.html)
- [Isaac ROS 版本说明](https://nvidia-isaac-ros.github.io/releases/index.html)
- [Isaac Sim 硬件要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [OpenMV 官方 USB 调试协议客户端参考](https://github.com/openmv/openmv/blob/v4.5.9/tools/pyopenmv.py)

## 验证与日志

```bash
source ./activate_project.sh
python -m unittest discover -s tests -v
```

协议检查覆盖 RGB565、灰度、JPEG、USB 分段读取和机械臂端口隔离。

GPU 管线使用独立 ROS domain 89 进行合成图像验证，以免测试图混入实时相机 GUI：

```bash
ROS_DOMAIN_ID=89 ./start_isaac_vision.sh
# 另一终端
source ./activate_project.sh
python tools/verify_isaac_vision.py
```

GPU 测试核对输出的 BGR 编码、尺寸和每个像素的通道顺序；测试图通过不等于 OpenMV 实体取流通过。真实摄像头还需单独验证 USB 识别、固件握手、连续帧和帧率。

GUI 日志 `work/openmv/gui.log`；GPU 验证结果 `work/isaac/gpu-conversion-check.json`；GUI 状态提示读取 `work/isaac/status.json`。

## 2026-10-19 验证结果

- GUI 已在本机 X11 桌面启动，显示 OpenMV 真实画面，可暂停/继续并保存当前图像。
- 先独立取得 10 帧真实 320×240 RGB 图像，再验证 GUI → ROS 2 → Isaac GPU 的完整实时链路。
- 独立 domain 89 的合成测试核对 3 帧，全部像素通过 RGB→BGR 通道转换检查。
- 实时链路配对 77 帧（按相同 ROS 时间戳），全部像素与预期 BGR 结果一致；采样约 4.15 秒，配对速率约 **18.32 FPS**，不是延迟或长期性能基准。
- 输入发布者与 Isaac 输入订阅者均为 RELIABLE QoS，已修复初次 BEST_EFFORT 与 Isaac 默认订阅不兼容的问题。
- 7 项 USB 协议检查通过。Isaac 安装新增 12 个包，未升级或卸载现有包。
- 已停止独立测试节点，保留实时 GUI 和 Isaac 图像节点运行。没有启用碗检测、手眼标定、机械臂运动或自动抓取。
- 实时验证细节：`work/isaac/live-camera-check.json`；输入 `/openmv/image_raw`，GPU 输出 `/isaac/openmv/image_bgr`。
