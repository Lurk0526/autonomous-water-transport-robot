> 本目录已接入 AstraGO 仓库。迁移与快速运行先读 [START_HERE](START_HERE.md)；以下保留原 Jetson 环境和历史部署说明。

# Jetson 抓碗项目环境

## 循迹主导航、无标线备用与真实输入接入（2026-11-01）

运行 `./run_cart_sim.sh`，打开 `sim/line_free_cart/output/report.html`，查看黑/白/黄循迹主方案、丢线切换、无标线备用、曲线走廊绕障、误差故障及取放流程。真实RGB-D已录制并检查；`./cart_preflight.sh`只读检查相机、底盘反馈与输入状态，`./cart_capture.sh`支持带元数据的现场录制。本轮无需租GPU。真实车辆参数和底盘反馈尚未提供，外参、现场精度及真实抓放/洒水未验收。详见[算法接入说明](cart/README.md)。原入口 `run_line_free_sim.sh`继续可用，纯标准库备用可加`--backup-only`。

## 主相机道路与深度感知（2026-10-30）

正式道路 `best` 模型已部署到 Orin，实时语义识别、RGB/深度关联、TensorRT FP16、录制回放和故障恢复已完成台架检查。运行 `./open_road_perception.sh` 打开界面；`./road_perception.sh status` 查看状态。完整操作和装车后待测项见[道路感知前置说明](docs/road-perception-preinstall.md)。当前尚未装车，不发布运动指令；机械臂视觉链路保持独立。

满水碗搬运：当前以[方案复审 v0.2](docs/bowl-transport-review-v0.2.md)和[运行说明](sim/bowl_transport/RUN.md)为准。本机入口为 `python3 sim/bowl_transport/preflight.py`，检查流程故障、数值模型及奖励漏洞；训练租赁准入仍未满足。尚未完成机器人关节资产、物理液体或抓碗策略训练。

当前机械臂跟随使用 **v1.2.0 响应优化配置**，由 `config/bowl-follow.runtime.json` 持久保存。六关节均可在 P500～P2500 内从当前位置建立会话，新姿态自动测量跟随方向；缩小居中死区、缩短动作等待，并根据真实位置反馈补偿小动作不响应。目标暂时丢失后，可自动重新锁定画面中连续出现的清晰目标碗。**v1.0.0** 初版及 2026-10-23 实机验收快照保存在 `releases/bowl-follow-v1.0.0.zip`；固定参数和原姿态模型仍位于 `config/bowl-follow.v1.json`、`config/pose-response-library.v1.json`。详见 [版本记录](CHANGELOG.md)。

## 碗轮廓标注

桌面双击 **碗轮廓标注**，或执行 `./start_annotation.sh`。工具为独立环境中的 Labelme 7.8.1，类别固定为 `target_bowl`，保留原图和 JSON。明天的标注步骤、负样本确认及 YOLO 分割导出命令见 [标注说明](docs/annotation-guide.md)。

## OpenMV 实时视觉与 Isaac ROS

2026-10-21 已部署训练后的 target_bowl 分割模型，GUI 显示 OpenMV 实时轮廓、置信度与像素中心，约 18 FPS。真实背景中发现支架误检，识别质量仍需补负样本验证。启动、停止与检验记录见 [部署记录](docs/bowl-deployment-20261021.md)。

后续已按用户要求暂停实时服务，并加入 [随碗移动的图像跟随控制](docs/bowl-follow.md)。默认只预览控制计划；实体方向、限位、识别准入与 CH340 复位问题尚待验证，不会随视觉一键启动驱动机械臂。

21 日晚已完成 [CH340 串口复位修复和实体试动](docs/arm-serial-fix-20261021.md)，重新连接不再触发已观测的启动回中广播。004 旋转与 003 低头已有位置反馈，用户确认看到了低头；实时视觉已恢复，运动跟踪等待目标碗与局部校准。

机械臂无法连接或不动时，可运行 `./.venv/bin/python tools/check_arm_connection.py` 检查控制板与六路舵机反馈，检查不发送运动目标。23 日的[总线诊断与启动提示修复](docs/arm-control-repair-20261023.md)记录了只有底座返回反馈时的排查证据。

重启或更换机械臂姿态后可直接点击“开始/继续跟踪”：程序读取六路当前位置，匹配可用的姿态模型，无需摆回昨天的位置。当前姿态与保存模型差异较大时，在当前位置自动做小幅方向测量；此时需要将碗放稳。肘腕继续采用渐进交接与平滑续接。

用户于 2026-10-25 明确要求全部关节范围可启动并确认安全，因此当前配置使用六关节 P500～P2500 全范围。
启动检查、运动规划和自动标定共用这一范围；到达端点时仅允许向范围内运动。
范围配置和验证情况见 [全范围启动](docs/full-range-follow-20261025.md)。
速度、灵敏度和当前实机记录见 [响应优化](docs/responsive-follow-20261025.md)。
本机 ROS 通信默认走回环接口，避免切换 Wi-Fi 后图像和识别中断；自定义 `CYCLONEDDS_URI` 或 `RMW_IMPLEMENTATION` 仍优先。
本次复发原因与验证记录见 [2026-10-25 跟随排查](docs/follow-repair-20261025.md)。

一键启动相机 GUI、GPU 分割与 Isaac 图像转换（可重复执行）：

```bash
./start_bowl_vision.sh
./start_bowl_vision.sh status
# 结束使用时：
./start_bowl_vision.sh stop
```

已接入机械臂自带 OpenMV H7 Plus（固件 4.5.9），本机 GUI 显示 320×240 实时画面，ROS 2 图像已接入 Isaac ROS 4.6 的 GPU 图像格式转换节点。当前仅启用视觉，没有自动抓取。

仅需要原始采图与 Isaac 时，也可分别在两个终端启动（不要与一键启动的 GUI 重复运行）：

```bash
./start_vision_gui.sh
./start_isaac_vision.sh
```

摄像头需用其自身的 Micro-USB 接口连接 Jetson。GUI 取流会临时暂停 OpenMV 原脚本，不覆盖板上 `main.py`。连接、话题、测试记录和 Isaac Sim 后续方案见 [OpenMV 与 Isaac 接入](docs/openmv-isaac.md)。

当前设备：Orin Nano Super，Ubuntu 24.04.4，L4T 39.2.1，系统 CUDA Toolkit 13.2，Python 3.12.3，ROS 2 Jazzy。

## 使用

```bash
cd "<项目目录>"
source ./activate_project.sh
```

实体环境为 `.venv-orin`，`.venv` 是指向它的兼容入口。激活脚本加载 Jazzy、项目 Python 和私有 CUDA 运行库。`deactivate` 会恢复原有 `LD_LIBRARY_PATH`；不修改 `.bashrc` 或全局 CUDA 链接配置。

## 当前组件

| 组件 | 配置 |
| --- | --- |
| PyTorch / torchvision | Jetson AI Lab 原生 Orin 2.8.0 / 0.23.0，`sm_87` |
| Torch CUDA 应用运行库 | 12.9，位于项目环境；系统驱动及 nvcc 仍为 13.2 |
| cuDNN / TensorRT | 复用 JetPack cuDNN 9.20 / TensorRT 10.16.2.10 |
| Ultralytics | 8.4.178 |
| NumPy / 项目 OpenCV | 1.26.4 / 4.11.0；系统 OpenCV 4.6.0 保留 |
| ONNX / ORT / ONNX Slim | 1.23.2 / 1.31.0（CPU）/ 0.1.98 |
| Orbbec Python SDK | pyorbbecsdk2 2.0.18，导入名 pyorbbecsdk |
| ROS | Jazzy；MoveIt、MoveItPy、Servo、ros2_control、控制器、Orbbec、Nav2/MPPI、TF、rosbag、图像与标定组件 |

## 两条警告的修复

此前通用 PyTorch 2.12.1+cu132 不包含原生 sm_87；其 cuSPARSELt 13 依赖还有错误的发行平台标签。现已切换到经过校验的 Orin 原生构建和匹配的 torchvision，补齐私有 CUDA 12.9 库、nvJPEG 以及 OpenBLAS。原生构建不依赖那个有问题的 cuSPARSELt 13 包，因此新环境没有它。

没有屏蔽警告、修改 PyTorch 架构判断或篡改供应商 WHEEL 标签。`pip check` 已输出 `No broken requirements found.`，严格检查将 UserWarning 转为异常后，初始化和 JPEG 编解码仍通过。

旧环境保留在 `work/fix-warnings/venv-before-warning-fix`，用于需要时恢复；当前项目入口使用新环境。

## 验证

```bash
python tools/verify_warning_fix.py
python tools/verify_project_env.py
python tools/verify_torch_tensorrt.py
python tools/verify_ros_env.py
```

已通过：

- 原生架构 sm_87、依赖/平台标签检查、JPEG 编解码。
- GPU 矩阵、卷积、NMS、反向传播、CPU/GPU 卷积数值比对、GPU 线性方程求解。
- 随机权重 YOLO11n 分割前向及输出有限值检查；合成图像为 160×160，不代表精度或性能验收。
- TensorRT 真实构建和 GPU 推理，直接使用 PyTorch 分配的 GPU 缓冲区与非默认流；验证系统 TensorRT 和项目 Torch 运行库的协同。
- ONNX 模型校验与 CPU 推理、cv_bridge 图像往返转换、MoveItPy、原生消息接口、本机 DDS 收发和 ROS CLI。

验证不访问相机、串口或机器人控制器。ROS 验证短暂创建本机 domain 88 的专用测试节点，结束后销毁。

项目 OpenCV 支持 V4L2/FFmpeg，不支持 GStreamer；系统 OpenCV 的 GStreamer 构建仍保留。相机准确型号、MV4 厂家驱动、URDF、关节限制、四爪和工具坐标还需资料与设备验证。

## 新建相同环境

仅适用于本项目已验证的 Jetson 系统栈。以下用于新建环境，不用于覆盖正在运行的环境：

```bash
sudo apt-get install libopenblas0-pthread
python3 -m venv --system-site-packages .venv-orin
.venv-orin/bin/python -m pip install -r requirements-jetson.lock.txt
python3 tools/configure_cuda_runtime.py --venv .venv-orin
ln -s .venv-orin .venv
source ./activate_project.sh
```

原生 Torch/torchvision 的来源和 SHA256 已写入 requirements，防止重建时误装同版本的通用构建。`requirements-cuda129.txt` 记录所需应用库；`constraints-jetson.txt` 保留 NumPy、setuptools 等兼容性限制。

## 文件与日志

- `tools/check_jetson_env.py`：通用环境诊断，默认不打开相机/串口。
- `requirements-jetson.txt` / `requirements-jetson.lock.txt`：当前依赖与版本锁。
- `ros-packages.txt`：ROS 和共享原生依赖清单。
- `ros2_ws/src/`：后续厂家驱动和项目 ROS 包的位置。
- `work/fix-warnings/`：本次修复、校验与回退记录。
- `work/`：ROS 日志、rosdep 缓存、视觉配置和其他诊断记录，不提交 Git。

来源：[Jetson AI Lab Orin Torch](https://pypi.jetson-ai-lab.io/jp6/cu129/torch/)、[torchvision](https://pypi.jetson-ai-lab.io/jp6/cu129/torchvision/)、[NVIDIA CUDA 12.9.1 组件清单](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json)。
