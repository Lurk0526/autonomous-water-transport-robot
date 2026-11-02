# AstraGO：本机工程运行入口

此目录从 Jetson 工作工程整理而来。所有命令都先进入 `software/jetson/`；不需要把已有工程拆成多个互相失联的目录。

本次通过网页导入文件，shell 文件的可执行位不会随上传保留。克隆或解压后，在本目录执行一次 `chmod +x ./*.sh`，以便原脚本之间的直接调用正常工作；仿真的 Python 入口不需要此步骤。

## 无硬件仿真

Python 3.10+。完整循迹套件需要 NumPy/OpenCV；没有这些依赖时，可先运行标准库备用场景：

```bash
cd software/jetson
python3 -m sim.line_free_cart.run --backup-only
```

在独立环境安装完整循迹依赖后运行：

```bash
python3 -m venv .venv-sim
source .venv-sim/bin/activate
python -m pip install -r requirements-sim.txt
python -m sim.line_free_cart.run
```

结果在 `sim/line_free_cart/output/report.html`。若下载 ZIP 后 shell 文件执行权限未保留，可用 `bash run_cart_sim.sh`，或直接使用上面的 Python 命令。

抓放任务监督器、数值筛查与奖励审计只用标准库：

```bash
python3 sim/bowl_transport/preflight.py
```

该入口生成 `sim/bowl_transport/output/report.html`；它不运行真实机器人、Isaac 或物理液体。

## Jetson 真实视觉

环境与原始部署记录见 [README](README.md)、[Gemini MAX](docs/gemini-max-ros2.md)、[道路感知](docs/road-perception-preinstall.md)、[OpenMV](docs/openmv-isaac.md)。原机器记录：Orin Nano Super、Ubuntu 24.04.4、L4T 39.2.1、Python 3.12.3、ROS 2 Jazzy；这些是原设备记录，不是任意 Jetson 的通用安装组合。

先恢复与配置匹配的模型和相机驱动工作空间，再启动视觉。`.venv`、Gemini 驱动工作空间、用户级服务、udev 权限与模型文件不随源码自动安装。部分相机脚本及服务模板仍引用原机器的 `/home/nvidia/gemini_ws` 或项目路径，迁移机器时须按实际位置调整，不能直接照搬服务模板。

```bash
source ./activate_project.sh
bash start_bowl_vision.sh       # 视觉入口；跟随运动另有明确入口
bash open_road_perception.sh    # 需恢复道路部署与 Gemini ROS 驱动
```

模型索引见 [models/](../../models/README.md)，数据说明见 [data/](../../data/README.md)。跟随配置含原机械臂的 P500～P2500 范围与实机标定记录；不能视为其他机械臂的通用关节配置。

## 当前约束

主方案兼容黑/白/黄单条赛道线，无标线导航仅作备用。真实轴距、制动、转角映射、相机到车体外参和 STM32 底盘协议仍待接入；测量表的未知项保留为空。

道路模型离线测试、RGB-D 台架采集、导航仿真与机械臂跟随分别记录，均不能替代实车自主取放、整车避障、30 km/h 或满水不洒的验收。
