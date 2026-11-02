# Gemini MAX 主摄像头环境

本机：Jetson Orin Nano Super，Ubuntu 24.04，L4T R39.2.1，ROS 2 Jazzy，ARM64。
任务背景：试验场西北角的室外砖块路段，覆盖直线、颠簸、转弯、避障和加速。
Gemini MAX 是车身主相机，用于道路感知、定位建图和避障；机械臂相机另行承担碗定位。
已确认的完整背景见 `docs/water-cart-project-context.md`。
相机：SV1301S_U3 / Gemini MAX，序列号 `AY2816200M0`，固件 `RD3013`。
深度接口 `2bc5:0614` 使用 USB 3.0 / 5000M；彩色接口 `2bc5:0511` 使用 USB 2.0，这是设备的两个不同接口。

## 启停与检查

双击桌面「Gemini MAX · RViz 相机调试」，或在本项目目录运行 `./open_gemini_rviz.sh`，
即可启动相机服务和 RViz。启动脚本会为 `systemctl` 单独选择 `/run/user/$(id -u)/bus`，
避免 VNC 独立会话总线导致 `Process org.freedesktop.systemd1 exited with status 1`、
RViz 窗口尚未打开便退出；RViz 本身保留当前桌面的显示和会话环境。

已提供用户服务 `gemini-max.service`。登录桌面后自动运行；没有设置无人登录时的系统级自启动。

```bash
systemctl --user status gemini-max
systemctl --user stop gemini-max
systemctl --user start gemini-max
journalctl --user -u gemini-max -n 80 --no-pager
```

终端使用 ROS 2 工具前加载本项目的环境（相机与订阅端需使用一致的 DDS 配置）：

```bash
cd <项目目录>
source ./activate_gemini.sh
/usr/bin/python3 tools/check_gemini_camera.py --seconds 15
rviz2 -d config/gemini-max.rviz
```

手动运行时先停止服务，再执行 `./start_gemini_camera.sh`；启动脚本通过文件锁防止重复占用相机。
修改 `config/gemini-max.yaml` 后执行 `systemctl --user restart gemini-max`。
`systemctl --user disable --now gemini-max` 可停止并关闭登录自动启动。
启动脚本会等待 `/dev/gemini-color` 出现且可读写后再运行节点，避免登录启动早于 USB 初始化，
导致彩色节点记录 `Failed opening device ... No such file or directory` 后一直无图像。

检查结果在 `work/gemini/health.json`，彩色预览在 `work/gemini/rgb.jpg`，
深度预览在 `work/gemini/depth_preview.jpg`；`depth_mm.png` 保存以毫米为单位的 16 位原始深度，0 表示无效像素。
ROS 启动日志同时保存在 `/home/nvidia/.ros/log/`；若系统 journal 未保存用户日志，可查看该目录。
本次实际帧率、重启和几何检查结果见 `docs/gemini-max-setup-report-20261029.md`。

## 数据接口

| 话题 | 类型 | 用途 |
|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/Image`，RGB8，640×480，目标 30 Hz | 路径识别的彩色输入 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 彩色内参 |
| `/camera/depth/image_raw` | `sensor_msgs/Image`，16UC1，640×400，目标 30 Hz | 毫米深度；未对齐到 RGB |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image`，16UC1 | 经相机内参矫正后的深度 |
| `/camera/depth/camera_info` | `sensor_msgs/CameraInfo` | 深度内参，与图像同时间戳、同坐标系 |
| `/camera/depth/points` | `sensor_msgs/PointCloud2`，XYZ 米 | 三维避障输入；有订阅时生成 |
| `/camera/scan` | `sensor_msgs/LaserScan`，米 | 深度图中央 10 行生成的相机相对扫描 |
| `/tf_static` | `tf2_msgs/TFMessage` | 相机内部坐标变换 |

深度和点云采用 SensorData / Best Effort QoS，彩色驱动使用默认 QoS。Python 订阅建议使用
`rclpy.qos.qos_profile_sensor_data`；`cv_bridge` 可转换为 OpenCV 图像。
ROS 使用系统 Python `/usr/bin/python3`，已有 `rclpy`、`cv_bridge`、OpenCV 4.6、NumPy 1.26。
需要 GPU 推理时，先 `source ./activate_project.sh`，再 `source ./activate_gemini.sh`，随后使用 `python`。
已验证现有项目的 PyTorch 2.8.0 / CUDA runtime 12.9 能在 Orin 上执行 CUDA 张量运算。
彩色使用系统 `v4l2_camera` 经原生 UVC 读取 YUYV，再发布 RGB8，绕开旧 SDK MJPEG 损坏帧问题。
`config/gemini-max-color-calibration.yaml` 保存了从这台相机读取的出厂彩色内参。
深度 SDK 不启用彩色，也不设置彩色曝光，避免抢占 UVC 接口。
直接使用 `cv2.VideoCapture(0)` 会与彩色 ROS 节点抢占接口，路径识别程序应订阅 ROS 图像话题。

配置中的扫描处理窗口为 0.3–5.0 米，不代表已标定该范围的测距精度。
中央窄条扫描不能覆盖完整三维空间；安装后的避障应使用点云并按地面高度过滤。

## 车体集成仍需的实际参数

用户已确认相机仍在调试、尚未固定。因此当前只发布相机内部 TF，不假设 `base_link → camera_link`，
暂不要求安装参数。最终安装后再测量相对车体的前后、左右、离地高度和姿态，将点云放到车体或地图坐标系。
手动启动可传入测量值，例如参数名如下（角度单位为弧度，位置单位为米）：

```bash
./start_gemini_camera.sh publish_mount_tf:=true \
  mount_x:=实际值 mount_y:=实际值 mount_z:=实际值 \
  mount_roll:=实际值 mount_pitch:=实际值 mount_yaw:=实际值
```

正 pitch 表示绕车体 Y 轴的右手旋转，在相机朝前的安装方式下通常对应向下俯视。
若机器人 URDF 已发布这条 TF，应由 URDF 维护，避免重复发布。

`config/gemini-nav2-observation.example.yaml` 是已有 Nav2 局部代价地图的点云观测源合并示例。
Nav2、深度转扫描、图像处理和可视化软件包已存在于系统。示例不能替代机器人完整导航配置；
还需底盘尺寸、里程计、TF、地图或定位来源及控制接口。
本次启动链不发布底盘速度，也不驱动机械臂。路径识别按已确认的室外砖路场景推进，重点是可通行区域、道路边界和障碍物；
没有将任务定义为黑线/色线循迹。需用这台车载相机采集现场数据，再验证对应感知算法。
首次实测彩色画面倒置，当前属于未固定的调试状态，未单独翻转 RGB；最终安装后统一校正相机姿态、图像与内参，保持深度和 TF 一致。

## 驱动来源、隔离与复现

这台设备使用 OpenNI 协议。系统已安装的 `orbbec_camera 2.9.3` 使用 SDK v2，未覆盖或卸载。
本项目使用随相机提供的 `ROS2/ROS2源码/orbbec_ws.zip` 中的 ARM64 SDK 1.5.8 和驱动源码，
将包改名为 `gemini_camera` / `gemini_camera_msgs`，单独编译到 `/home/nvidia/gemini_ws`。
ROSIDL 在中文工作空间路径下生成消息失败，因此构建使用上述英文路径。

兼容修改保存在 `drivers/gemini-max-jazzy.patch`：Jazzy 参数回调类型、SDK v1 头文件和库的优先级、
CameraInfo 时间戳/尺寸/畸变系数顺序、静态 TF 平移、停止时的线程空指针检查、较小的 SDK 队列、
禁用流的传感器访问隔离、序列号选择和初始化失败后的重试。
未升级相机固件。相机访问规则在 `/etc/udev/rules.d/99-gemini-max.rules`，仅开放该型号的两个接口给现有 `video` 用户组。
源码归属和许可沿用供应商，未上传到外部服务。

`config/cyclonedds-gemini.xml` 为本机回环链路配置较大的 RTPS 分片和接收缓冲区。
`/etc/sysctl.d/90-gemini-dds.conf` 将收发缓冲区上限设为 32 MiB；各 socket 按需分配，
避免默认约 208 KiB 上限导致同时订阅 RGB、深度、点云时大量丢包。现有机械臂项目的 DDS 配置未修改。
参考：[Cyclone DDS 配置说明](https://cyclonedds.io/docs/cyclonedds/latest/config/config_file_reference.html)。

源包 SHA256 和补丁记录见 `/home/nvidia/gemini_ws/source-provenance.json`。
如需在本机重新生成源码，保留现有工作空间，选择另一个英文目录：

```bash
/usr/bin/python3 tools/prepare_gemini_driver.py --workspace /home/nvidia/gemini_rebuild_ws --build
```

官方协议和分支说明：[Orbbec SDK v1](https://github.com/orbbec/OrbbecSDK/tree/main)、
[Orbbec ROS 2 驱动](https://github.com/orbbec/OrbbecSDK_ROS2)。
