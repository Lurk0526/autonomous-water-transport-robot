# Windows 标注与训练交付包

适用目标：Windows 11 x64、RTX 4060 8 GB、Python 3.12 x64。

第一版模型：YOLO11n-seg；唯一类别：`target_bowl`。模型在电脑上训练，OpenMV 负责采图。当前没有已训练好的专用碗模型，包内的 `.pt` 是通用预训练权重。

## 先安装哪些环境

| 项目 | 安装要求 |
| --- | --- |
| 系统 | 推荐 Windows 11 64 位，包内脚本使用 Windows CMD |
| 显卡驱动 | 安装适配 RTX 4060 的当前 NVIDIA 驱动；命令提示符运行 `nvidia-smi` 应能显示显卡 |
| Python | 安装官方 64 位 Python 3.12，包含 pip/venv 和 Python Launcher；`py -3.12 --version` 应成功 |
| VC++ 运行库 | 安装 Microsoft Visual C++ Redistributable x64 |
| 内存/空间 | 建议至少 16 GB 系统内存、30 GB 空闲磁盘，另留照片与训练结果空间 |
| 网络 | 首次安装需要联网下载 Windows 依赖和 PyTorch；这个包不是完整离线安装包 |

Python 官方新安装管理器如支持，可用 `py install 3.12` 安装该版本；如果已有 Python，先用 `py -0p` 检查版本。不要用 32 位 Python。

PyTorch 使用官方 Windows 预编译 CUDA 12.8 版本：`torch==2.8.0`、`torchvision==0.23.0`。安装脚本会装好 Python 运行依赖；本流程无需先安装完整 CUDA Toolkit、cuDNN、Visual Studio、ROS 或 Isaac。NVIDIA 驱动仍需正常安装。

参考安装入口：

- [Python Windows 下载](https://www.python.org/downloads/windows/)
- [NVIDIA 驱动](https://www.nvidia.com/en-us/drivers/)
- [Microsoft VC++ x64 运行库](https://aka.ms/vc14/vc_redist.x64.exe)
- [PyTorch 2.8 官方版本组合](https://pytorch.org/get-started/previous-versions/#v280)

## 解压与安装

把整个 ZIP 解压到例如 `C:\jibot-bowl\`，保留内部目录结构，不要在 ZIP 预览窗口里直接运行。优先用短路径，避免 OneDrive 同步目录。

依次双击：

1. `windows\setup_annotation.cmd`：创建 `.venv-labelme`，安装 Labelme 7.8.1。
2. `windows\setup_training.cmd`：创建 `.venv-train`，安装 GPU 训练依赖，并实际检查 CUDA 张量、卷积反向传播和 torchvision NMS。

两个环境独立，避免标注软件与训练依赖互相影响。首次安装耗时取决于网络；看到错误后不要关闭窗口，保留输出便于排查。环境检查记录在 `work\training-env.json`。

包中带了 Labelme 和 Ultralytics 的通用 Python wheel 及源码归档，但 Qt、PyTorch 等平台依赖由安装脚本联网下载。Jetson 的 `.venv` 和 ARM64 CUDA 库没有打入包中，也不要在 Windows 安装 `requirements-jetson.txt`。

## 图像与标注

自行采集的照片和原有同名 JSON 请一起复制。本包不包含完整训练数据；数据目录是空模板。

```text
datasets/target_bowl/source/
  train/拍摄批次A/照片.png、照片.json
  val/拍摄批次B/照片.png、照片.json
  test/拍摄批次C/照片.png、照片.json
```

按拍摄批次分约 80%/10%/10%。同一段视频的相邻帧放在同一划分里，每个划分至少保留目标碗正样本，并加入其他碗、黑色杯盘、空桌面等对照。

双击 `windows\start_annotation.cmd`：

- `Ctrl+N` 创建多边形；一只目标碗对应一个实例，类别选 `target_bowl`。
- 标属于碗的可见像素：碗身、碗沿、可见内壁和碗底。排除手、食物与背景。
- 双击闭合，`Ctrl+S` 保存，`D` 下一张，`A` 上一张。自动保存也已开启。
- 无目标碗时勾选图像标志 `negative` 并保存；未标注图不能当负样本。
- 当前导出器不支持孔洞或同一个实例的多个分离区域。复杂遮挡图先留待复核，不要错误填满轮廓，也不要把一只碗拆成多个实例。

## 数据检查与导出

双击 `windows\export_dataset.cmd`。它先检查标注，全部通过后生成：

```text
datasets/target_bowl/yolo/
  data.yaml
  images/train、val、test
  labels/train、val、test
  manifest.json
```

仅检查不导出：在包根目录的 CMD 中运行 `windows\export_dataset.cmd --check-only`。

已有导出目录时不会覆盖；新版本可运行：

```bat
windows\export_dataset.cmd --output datasets\target_bowl\yolo-v2
```

原图、原标注不改动。导出会检查尺寸、标签、轮廓和跨划分重复图片。JSON 与中文路径按 UTF-8 读取。

## 训练与评估

数据就绪后双击 `windows\train.cmd`。RTX 4060 的初始配置为：320 输入、batch=8、80 轮、workers=0、单卡训练、FP32。参数是起点，不是精度或速度保证。显存不足时在 CMD 中运行：

```bat
windows\train.cmd --batch 4
```

更换数据导出目录：

```bat
windows\train.cmd --data datasets\target_bowl\yolo-v2\data.yaml
```

训练结果位于 `runs\segment\target-bowl-v1\`；若已有同名结果，Ultralytics 会使用新的编号目录。实际路径会打印到窗口并记录在 `work\last-training.json`。保留 `weights\best.pt` 和 `weights\last.pt`。

训练使用 val 集选择模型。完成后用独立 test 集检查误检、漏检和 mask 质量：

```bat
windows\validate.cmd
```

默认查找最新一次训练的 `best.pt`；有多个模型时可明确指定：

```bat
windows\validate.cmd --weights runs\segment\target-bowl-v1\weights\best.pt
windows\predict.cmd --source "C:\test-images" --weights runs\segment\target-bowl-v1\weights\best.pt
```

不要一边看 test 结果一边反复调参；调参用 val 集，最后再测 test。

## 返回 Jetson

将 `best.pt`、`last.pt`、`results.csv`、训练参数及评估结果一起带回。数据搬家后，`bowl_model.py` 会生成当前机器专用的 YAML 路径配置，不修改原 `data.yaml`。

需要 ONNX 时运行 `windows\export_onnx.cmd`，依赖会按需安装。TensorRT `.engine` 另在 Jetson 上构建，Windows 生成的 engine 不作为 Jetson 部署文件。

## 源码与许可

- `tools/`：数据校验导出、训练、验证、预测与环境检查源码。
- `vision/`、`launch/` 和 Linux 启动脚本：现有 OpenMV/ROS/Isaac 端源码，供 Jetson 端继续开发；不属于 Windows 训练前置依赖。
- `third_party/`：Labelme 与 Ultralytics 官方 wheel、源码包和许可证说明。
- `reference_firmware/`：原厂 STM32 源码归档和 OpenMV 例程，作为参考；无需为训练刷写设备。
- 包内 `MANIFEST_SHA256.json`：文件清单和校验值。

可编辑标注工具上游源码在 `third_party/labelme-7.8.1.tar.gz`；训练框架上游源码在 `third_party/ultralytics-8.4.178.tar.gz`。保留对应许可与第三方声明。

当前宿主是 Jetson Linux。本交付会验证 Python 源码、标注导出和训练流程；Windows CMD 启动与 RTX 4060 实机性能需要在你的 Windows 电脑执行检查后确认，不能视作已在 Windows 实测。

官方参考：[Labelme](https://github.com/wkentaro/labelme)、[Ultralytics 训练](https://docs.ultralytics.com/modes/train/)、[分割格式](https://docs.ultralytics.com/datasets/segment/)、[TensorRT 平台限制](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)。
