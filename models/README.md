# 模型索引

已从 Jetson 工程补充实际文件的大小与 SHA-256：[artifacts-20261102.json](artifacts-20261102.json)。权重二进制、ONNX 和 TensorRT plan 按仓库原约定保留在原工作工程/离线交付存储，未配置公开下载或 LFS。克隆源码后不能直接运行依赖这些权重的真实推理。

| 模型 | 记录 | 恢复位置（相对 `software/jetson/`） |
| --- | --- | --- |
| target_bowl YOLO11n-seg | `target-bowl-20261021`，单类碗分割；[部署记录](../software/jetson/docs/bowl-deployment-20261021.md) | `work/models/target-bowl-20261021/best.pt` |
| 道路 SegFormer-B0 | `road-4060-v1`，20 轮训练后选第 8 轮 best；7 类、384×512；[部署记录](../software/jetson/docs/road-perception-preinstall.md) | `work/models/road/road-4060-v1/` |
| 道路 ONNX / TensorRT FP16 | 原机导出/数值比较记录；engine 需在匹配的目标平台重建 | 同上 `accelerator/` |

道路原有测试输出保存在 [results/runtime-import-20261102/road-model/](../results/runtime-import-20261102/road-model/)。六类 mIoU 79.62% 为排除水类别后的离线平均，铺装 IoU 94.41%；测试集仍含水面。既有 8 张验证图的 FP16/FP32 比较不等于全测试集精度或现场验收。

碗模型本次未补出缺失的独立场景 mAP，不以演示画面替代指标。完整数据及来源见 [data/](../data/README.md)。训练、评估、部署清单验证和 engine 构建程序都在 `software/jetson/tools/`。
