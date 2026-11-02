# Jetson＋OpenMV 目标碗部署与检验

日期：2026-10-21，Asia/Shanghai。当前目标机就是本项目所在 Jetson，无需 SSH。

后续状态：用户要求先暂停并编写随碗移动逻辑；三项实时服务现已停止。跟随代码、默认预览与实体启用前的标定条件见 [图像跟随说明](bowl-follow.md)。下文的实时性能来自暂停前的已完成验证。

## 结果

部署与实时链路通过；“只识别训练用目标碗”的场景准确性尚未通过验收。

- 用户提供的 best.pt 已复制到 `work/models/target-bowl-20261021/best.pt`，原附件保留。
- SHA256：`c402769dabe8f46fd69e01f76054622b15677862fd34e1825618fedc4b9073a6`，与进度手册一致。
- 检查模型为单类别 `{0: target_bowl}` 的 YOLO11n 分割，训练配置为 80 轮、640 输入、batch 4。Checkpoint 附带验证集指标不等于手册记录的独立 test 指标。
- Jetson 原生 Torch 2.8.0、Ultralytics 8.4.178 与 GPU 推理通过，沿用现有环境，未安装或升级依赖。
- OpenMV USB `1209:abd1`、`/dev/ttyACM0`，H7 Plus、固件 4.5.9，实际连续采图成功，尺寸 320×240。
- 单一 GUI 进程占用 OpenMV。只在 RAM 运行采图脚本，未覆盖 main.py，未打开机械臂 CH340 端口。

## 启动与停止

在项目根目录运行：

```bash
./start_bowl_vision.sh
./start_bowl_vision.sh status
./start_bowl_vision.sh stop
```

使用当前用户的临时 systemd 会话服务，重复启动不会再开一套采图进程；没有设置开机自动启动。GUI 显示原始画面或同一推理源帧的分割叠加，采图不会等待 GPU 推理。

三个服务为 `jibot-openmv-gui`、`jibot-bowl-detector`、`jibot-isaac-vision`。只停识别可执行 `systemctl --user stop jibot-bowl-detector`，恢复用 `./start_bowl_vision.sh`。

独立调参入口：

```bash
source ./activate_project.sh
# 先停已有推理服务，避免重复节点；保留 GUI 采图。
systemctl --user stop jibot-bowl-detector
./start_bowl_detection.sh --conf 0.5 --imgsz 640
```

默认 FP32、GPU 0、640 输入、置信度阈值 0.5。该阈值是部署起点，尚未通过独立负样本集选择；本次没有通过提高阈值隐藏误检。没有导出 TensorRT。

## 数据流与输出

```text
OpenMV USB → GUI → /openmv/image_raw (RGB)
                    ├→ Jetson YOLO 分割 → /openmv/bowl/overlay (RGB)
                    │                 → /openmv/bowl/detections (JSON String)
                    └→ Isaac GPU 转换 → /isaac/openmv/image_bgr (BGR)
```

分割节点订阅已有原始图像，避免第二个进程占用 USB；目前模型直接使用原始图像，Isaac 转换分支同时保留并核验。JSON 包含原图时间戳、轮廓、掩膜像素面积、像素质心、置信度、推理耗时和帧率。NumPy 输入按 Ultralytics API 转为 BGR，输出叠加转回 RGB。掩膜质心只是图像统计量，凹形目标时也未必落在可抓取区域内。

订阅队列深度 1，消费最新可用帧；输入或推理完成后的源帧年龄超过 1 秒则丢弃。无新图像时发布 `valid=false` 并清空目标。GUI 同样不显示超过 1 秒的旧叠加。时间戳沿用 Jetson 接收/发布图像时刻，不是曝光时间，因此所测源帧年龄不包括完整相机曝光与 USB 传输延迟。

`work/bowl-live/status.json` 保存最新状态，判断时还需核对 `updated_unix` 的新鲜度；进程被强制杀死后磁盘文件不会自行更新。每 5 秒保存一组时间戳对应的 `latest-raw.png`、`latest-overlay.png`、`latest-sample.json`。场景有/无目标变化时限速保存示例，`events/` 默认累计最多 200 组。GUI 的保存按钮继续只保存原始图像。

## 本次验证

- 16 项现有测试通过，包括 USB 分包、RGB565/灰度/JPEG 解码、CH340 隔离及数据导出。
- 独立 ROS domain 90 的模型与消息链路检查通过：历史目标参考图置信度 0.9169，合成黑图零目标，停止输入后清空旧结果，叠加保留源图 header。合成黑图不代表真实负样本精度通过。
- 实际相机链路采样 12.01 秒，221 条分割结果、217 帧叠加，约 18.40 FPS；平均推理 42.24 ms，平均源帧年龄 48.92 ms，最大 87.12 ms。这是一次短时采样，不是长期性能基准。
- 209 对相同时间戳的真实相机帧逐像素核对 Isaac RGB→BGR 转换，全部一致。
- 已目视检查 GUI 显示实时图像、识别状态和 Isaac 当前输出，不再引用历史状态文件作为当前运行状态。
- 用户把碗拿入镜头后已产生真实轮廓，保存的例图置信度约 0.776、掩膜像素中心约 `(224.7, 95.3)`。
- 同时发现上方黑色支架误检，示例置信度约 0.644，中心约 `(73.5, 28.1)`。部分遮挡、边缘位置和衣物区域也出现不稳定候选，不能按候选数声称准确率。

证据在 `work/deployment-20261021/`：`live-verification.json`、`contract-verification.json`、`review/live-target-bowl.*` 与 `review/false-positive-support.*`。真实负样本错误必须保留，不能用这一轮现场片段评估完整独立测试精度。

本机未收到对应的 47 张独立 test 图片及标注，因此没有复算手册的 test mAP50 99.50%、mAP50-95 93.79%、Recall 100%；这些仍为用户提供的历史结果。

## 尚需完成的精度工作

补充并人工确认实际工作区、空桌面、黑色支架、衣物、相似杯子与其他碗等负样本，按拍摄组划分，再重训与独立验证。目标应放在实际桌面抓取视角，并覆盖反光、遮挡和目标移出画面的情形。

当前相机未标定，只有单目 RGB；尚无三维距离、手眼外参或抓取坐标。此次仅启动视觉识别，不包含机械臂运动。

API 参考：[Ultralytics 预测与 NumPy 颜色格式](https://docs.ultralytics.com/modes/predict/)。
