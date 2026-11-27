# Gazebo 一米环道 · Mac 训练与闭环验证

[Mac 原生闭环结果](../../../results/gazebo-ring-v3/README.md)。仓库提供源码、赛道、网格和选定验证结果。下文的离线安装需要完整原始包，视觉模型按 `models/manifest.json` 核对，离线依赖按 `payload/sources.json` 核对。

这套任务使用 Gazebo Harmonic 的轮地物理和车载 RGB-D 相机。相机图像经过项目的道路分割和雪糕桶检测 ONNX，驾驶策略预测目标车速（m/s）与偏航角速度（rad/s）。Gazebo 的阿克曼转向插件把命令作用在车轮上。专家可以查看赛道和车姿，用于生成示范标签；学习策略根据车载图像和对齐深度开车。独立评分器从模拟世界位姿、碰撞和有方向的测速门计算成绩。

## 赛道模型

选取了 **AWS DeepRacer Oval Track** 作为闭环赛道参考。包内有固定上游版本的视觉网格、碰撞网格、材质、原始 SDF、路点和 Apache-2.0 许可。读取原模型的路点，中心线长约 19.55 m。本项目沿用可独立检查的中心线、内侧线、外侧线三组路点表示，自行生成一米黄线走廊、纵向绕桶区、砖段和测速区。车辆参考 Gazebo Harmonic 的 AckermannSteering 示例，以后轮驱动和前转向节控制本项目车模。[原版赛道](https://github.com/aws-deepracer-community/deepracer-simapp/blob/15f62e094b3d6f6753d78671f7dba3520a9dc1ff/bundle/models/Oval_track/Oval_track_model.sdf) · [许可](references/deepracer-oval/LICENSE) · [车辆参考](https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/ackermann_steering.sdf)

要看比赛场景，安装 Gazebo Harmonic 后在包内运行 `gz sim worlds/ring.sdf`。用 `worlds/ring-waypoints.csv` 可查看中心和内外黄线坐标，第一行点与末点重合。白色虚线为共同起终门。权重训练与评测入口会为每一圈生成单独的场景与记录。[一米环道 SDF](worlds/ring.sdf) · [赛道几何校验](worlds/geometry-check.json) · [俯视布置和车载画面](validation/一米环道实录.png)

原始 Oval Track 和一米比赛版分别可以直接打开。下面是从包根目录打开原版的命令：

```bash
GZ_SIM_RESOURCE_PATH="$PWD/references/deepracer-oval/bundle" gz sim references/deepracer-oval/bundle/worlds/Oval_track.world
```

## 成品车 CAD

从最终版 `成品车.SLDASM` 引用的 `第三版底盘.SLDASM` 读取四个轮胎实例和保存的显示网格。模型里的轮胎外宽约 **78.7 cm**，前后轴距 **57.8 cm**，前后轮距约 **69.0 cm**，轮胎直径 **22.0 cm**。Gazebo 使用这个轴距和轮距，车架、亚克力板、电池、Orin 和四轮外观使用提取的网格。评分用保守的 **80 cm** 宽外廓，保留前后悬。显示网格覆盖 92 个部件实例，动力学使用简化的阿克曼底盘、圆柱轮胎与刚体连接。实车悬架、制动、质量和 Gemini MAX 外参留待实体标定。[尺寸与来源哈希](vehicle_assets/cad-measurements.json) · [Gazebo 中的 CAD 车模](validation/CAD车模Gazebo.png) · [采用的 SolidWorks 网格读取器](https://github.com/XRTC5/sldprt-export/tree/31d3b5a9b115d9cbfc19bbf10906fcc29e8e3c57)

## 在 Mac 上训练

解压完整 ZIP，将整个 `gazebo-ring-v1` 目录放在本机磁盘。下列命令在此目录执行。Apple Silicon Mac 使用原生终端。Python 3.12、适用 macOS arm64 的依赖、两个视觉模型及逐场特征已随包附带，初次安装脚本会校验 SHA256。

**用本机已采集的 Gazebo 圈训练**，双击 `Mac训练.command`，也可执行：

```bash
bash Mac训练.command
```

此入口离线拟合常量基线、岭回归和小型神经网络。按独立场次划分训练、验证、留出测试，在验证集选择控制策略。输出目录为 `outputs/mac-时间戳/`，其中 `training/policy.npz` 可加载驾驶，`training/metrics.json` 给出各组误差和种子，`workflow.json` 记录实际命令、日志和退出状态。动作预测误差用于比较拟合效果；能否避桶、完赛和通过测速需要闭环验证。纯 NumPy 策略训练使用 CPU。安装后运行 `.venv/bin/python tools/check_perception.py` 可用随包的左绕、右绕、无桶三对实采仿真 RGB-D，核对本机两个模型提取的特征与 Jetson 的差异。

**在 Mac 上自己开 Gazebo 采集、训练并回到赛道复试**，先安装 Gazebo Harmonic、CMake 和 Apple 命令行编译工具。已有 Gazebo 的用户可直接先运行版本检查。在已有 Homebrew 的环境中：

```bash
brew tap osrf/simulation
brew trust osrf/simulation
brew install gz-harmonic cmake
gz sim --versions
```

输出应含 8.x。若 `xcrun --find clang++` 报缺编译器，请执行 `xcode-select --install` 并完成系统安装。首次闭环将通过 CMake 编译小型 Gazebo Transport 桥接程序，连接本机隔离的仿真分区。Gazebo 的 macOS 安装说明明确为 Apple Silicon Sequoia 提供预构建包；你们的 macOS 27 可能需要 Homebrew 从源码构建。Python 离线训练可直接先跑。[Gazebo 官方 macOS 安装说明](https://gazebosim.org/docs/harmonic/install_osx/)

```bash
bash Mac训练.command --collect --episodes 16 --start-seed 100 --eval-episodes 4
```

该命令运行 16 次专家整圈（4 个种子，每个种子各含左绕、右绕、交错和无桶），保存相机图、以米为单位的深度、标签和每圈结果，再提取感知特征、训练并用另外 4 个种子运行带几何引导的学习策略。同一随机种子的四种布置整组进入同一数据划分。接电运行，原始深度逐帧保存，建议至少预留 3 GB。较慢的 Mac 优先运行默认的离线训练。单独测试已有特征新训练出的策略，可用 `bash Mac训练.command --closed-loop`。只回看闭环时，下列命令可直接载入已有的 `policy.npz`。请替换策略路径和结果目录。

```bash
.venv/bin/python -m ring_training.gazebo_run run --controller guided-policy \
  --policy outputs/某次训练/training/policy.npz --layout slalom --cone-count 5 \
  --seed 2000011 --camera-hz 12 --output outputs/检查一圈
```

闭环评分出现漏门、未达速、碰撞或越界时，打开该次的 `result.json` 及 `truth_trace.csv`。若未能启动或没有可用 RGB-D，检查同目录的 `gazebo.log` 和 `transport.log`。若 Ogre2 的 RGB-D 渲染在 Mac 上失败，可先用 `smoke --render-engine ogre` 检查，再给 `Mac训练.command` 加 `--render-engine ogre`。macOS 服务端渲染需要图形会话。每个采集与评估命令都使用一个新目录。

## 赛道和判定

- 顺行驶方向摆 **3–5 个**雪糕桶，每个桶单独占一个纵向位置，间距首版约 5.2 m。示范包含同侧连续桶、左右交错桶和无桶。车应逐个绕过再回线。中心堵塞另设为停车反例。标准桶底座宽 34 cm，摆位和间距可重采验证。
- 这里的体育场形中心线约 251.4 m。两条黄色边线分别沿内环和外环闭合。线中心距 **106 cm**，每条线宽 **6 cm**，两线内缘之间净宽 **100 cm**，铺装总宽 **144 cm**。正常走两线之间，线外每侧留出 **16 cm** 铺装供绕障。只在有桶的绕行段允许轮胎压线，按轮胎接地宽度分别记录内线、外线，按全车轮廓与铺装外缘判硬越界。图中的实际可用宽度、尺寸和标线布置需在实体阶段测量。
- 砖段长 14 m，有凸起碰撞几何和砖面，前、后两道门依次通过才算完成。外部路沿、地面高差、车体和路面摩擦均为模拟参数。
- 测速门间距 20 m，按西向穿过。蓝色标记预告加速，青色标记进区，紫色标记出区。这些颜色是让相机能辨认分区的仿真约定，实地需换成现场有的标志或来自已定位赛道的分区信息。30 km/h 仅适用于测速区。
- 一圈须按顺序通过起点、砖段与测速门，足程返回起点。全车车身与桶底座重叠或发生接触、跑出外缘、逆行均计失败。观察每圈的 `speed_zone_min_kmh`、细采样最小地速与门距/时间均速；目标油门或峰值速度不能代替入区后的最低实速。
- 根目录 `models/manifest.json` 固定模型来源与哈希。当前仿真相机最远深度约 24 m，台架道路程序处理窗口为 5 m；真实路况的深度空洞与振动还需实测。已训练锥桶检测器在独立 AHRV 车载远景的 38 个标注框只检出 2 个，仿真阶段没有修复这个现场感知缺口。简单车轮、制动、相机安装和载荷也有待实体标定。

## 看本机实测

在 Jetson 上用 Gazebo Harmonic 8.15.0 完成 16 圈专家示范，采集 3,205 对 RGB-D，从中提取 1,605 个带动作标签的样本。按种子隔离训练、验证和留出场景，留出集车速预测平均绝对误差约 0.324 m/s，偏航角速度约 0.0136 rad/s。

**冻结策略后，几何引导加学习微调通过了以下 4 圈未参与训练的赛况。** 所有圈通过 14 m 砖段并回到同一门，无碰桶、非法压线或越出铺装边界。测速值为整个区间相邻模拟位姿计算的最低地速。

| 布桶 | 计圈时间（s） | 测速区最低（km/h） | 触线 |
| --- | ---: | ---: | --- |
| 3 桶，同侧连续左绕 | 74.11 | 35.26 | 内黄线 |
| 4 桶，同侧连续右绕 | 75.10 | 35.10 | 外黄线 |
| 5 桶，左右交错 | 76.14 | 34.99 | 内、外黄线 |
| 无桶 | 70.55 | 35.15 | 无 |

引导层借鉴 Pure Pursuit，使用 RGB-D 恢复走廊、跟踪连续锥桶距离并规划换侧，根据砖面和测速标志调整速度。模型微调在四圈各生效 807–908 帧。相同的 5 桶场景里，独立的几何视觉控制也通过，单独运行学习策略会碰到第一个桶。居中桶堵路反例停在桶前，实测车速回到 0，计分未完赛。图像特征重跑、离线训练入口和 Apple Silicon 依赖解析已在本机验证，Mac 实机执行待复核。[Pure Pursuit 参考](https://github.com/ros-navigation/navigation2/tree/main/nav2_regulated_pure_pursuit_controller)

[完整验证记录](validation/local-results.json) · [Gazebo 相机和赛道布置](validation/一米环道实录.png)

在本机 Jetson 采用同一套模块，先加载 `/opt/ros/jazzy/setup.bash`，Python 环境需能导入 numpy、cv2、onnxruntime 和 PIL。独立重做动力学与传感器烟测可运行：

```bash
python -m ring_training.gazebo_run smoke --seed 0 --output outputs/smoke
python -m ring_training.gazebo_run record --seed 11 --layout slalom --cone-count 5 --sample-hz 2.5 --output outputs/演示11
```

实体连接后再实测车身尺寸、相机外参、制动距离和整链路延时。仿真策略的输出仅发往该次 Gazebo 分区。
