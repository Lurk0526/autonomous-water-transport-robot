# 室外砖路道路识别：训练方案

目标是试验场西北角的室外砖路，覆盖直线、颠簸、转弯、避障和加速。先用公开数据建立道路感知基线，再在实际 Gemini MAX 输入上验证。

## 模型分工

| 任务 | 模型或方法 | 当前用途 |
|---|---|---|
| 路面和边界 | SegFormer-B0，约 372 万参数 | 像素级语义分割，提供道路外观区域 |
| 人、车辆等常见物体 | YOLO11n 预训练检测 | 类别辅助；后续按误检漏检决定是否微调 |
| 碗轮廓 | 已有 YOLO11n-seg 和本项目碗权重 | 机械臂视觉链路继续复用 |
| 任意障碍物距离 | Gemini MAX 深度/点云 | 识别类别之外的几何占据和净空 |

SegFormer 用于整片路面分割。此前的 YOLO11n 系列保留用于物体和碗；不把普通实例分割权重当成已训练的道路模型。下载的 SegFormer 在 Cityscapes 上预训练，这提供初始化能力，尚未适配你们的砖路。

## 数据和划分

CamVid 来源于 GitHub `alexgkendall/SegNet-Tutorial`，源快照为 `bb68b64e72cb004b3ed927fee0f5624dd8097498`。共有 701 对图像和标注。原始划分有同一视频跨集合的情况，因此本项目按完整源视频重新分组：训练 406 张，验证 124 张，测试 171 张，原始划分字段仍写入清单便于追溯。这是本项目划分，成绩不能直接与使用原始划分的论文比较。

RUGD 由 GitHub `dataset-ninja/rugd` 的下载说明定位到原作者网站，下载完整 RGB 图像和像素标注。它含低位车载视角、植被、土路、铺装地面、阴影和模糊，较接近本任务的一部分挑战。使用论文 Table I 的完整场景划分：`park-8` 和 `trail-5` 验证，`creek`、`park-1`、`trail-7`、`trail-13` 测试，其他场景训练。实际标注归档检查到 7,436 张：训练 4,779，验证 733，测试 1,924；与论文给出的帧数略有不同，验收以本次归档实际文件为准。

两个数据集合并后的预期数量：训练 5,185，验证 857，测试 2,095。图像和标签须逐一配对检查，彩色标注只接受原始类别表中的颜色；所有掩码缩放使用最近邻插值。清单检查同一视频和相同图像哈希不能跨训练/验证/测试集合。

原始数据在 `datasets/public_road/`，合并清单和处理后标签在 `datasets/public_road/prepared/`。下载归档在 `work/road_training/downloads/`。最终验收报告记录已完成的下载和校验，不能仅凭 `.part` 文件的显示大小判断下载完成。

## 统一标签

训练七类：铺装地面、松散或粗糙地面、植被、水面、固定障碍物、人或车辆、天空。无标注区域使用 255 忽略。具体映射在 `config/road-labels.json`，不直接拿 RGB 标注值当类别编号。

铺装地面包含 CamVid 的 road/pavement 和 RUGD 的 asphalt/concrete，这表示外观类别。它仍可能包含不允许本车进入的人行区域；导航必须结合指定路线、实际道路边界和深度几何。砖路没有独立真值类别，公开数据无法直接给出“西北角砖路专用模型”的验收结论。

草地、土路、松散石料和水面不自动作为可通行区域。低置信度、无效深度和类别未知位置也不能直接当作空地。RUGD 的 bridge 标签未区分桥面和结构，暂保守归入固定结构。

## 分阶段训练

| 阶段 | 数据 | 设置 | 验收 |
|---|---|---|---|
| 下载与审计 | 两个完整数据集 | 文件完整性、图像/掩码配对、类别及划分 | 数据数量与来源报告 |
| GPU 小规模检查 | CamVid 少量训练/验证图 | 256×320，12 步，冻结骨干 | 有限 loss、反向传播、保存/加载 |
| A：铺装场景预热 | CamVid 全训练集 | 384×512，5 epochs，冻结骨干 | 验证集选择 best |
| B：室外联合训练 | CamVid + RUGD | 384×512，20 epochs，全网微调 | 分数据集指标与平衡选择分数 |
| 独立测试 | 固定测试清单 | 不参与训练和挑选权重 | 分类别 IoU、铺装精度/召回、错误可通行率 |
| 部署验证 | 固定权重和实机输入 | 先 ONNX 一致性，再本机 TensorRT FP16 | 实际总延迟、画面稳定性、现场错误案例 |

联合训练按约 25% CamVid、75% RUGD 抽样，避免前者在更大的 RUGD 中被淹没。AdamW，骨干学习率 `6e-6`，头部 `6e-5`，weight decay `0.01`；线性预热后多项式衰减。单次 batch=1，累积 4 次更新，AMP，workers=0，不缓存全数据。梯度裁剪 1.0，训练中遇到非有限 loss 或持续的非有限梯度会失败并保留已保存检查点；AMP 的偶发缩放溢出会记录、跳过该次更新并降低缩放系数。

小 batch 冻结解码头 BatchNorm 的运行统计。类别权重只根据训练集像素统计计算并限幅，不能使用测试集分布调整。增广包括左右翻转、亮度/对比度/颜色变化、轻微模糊；不做上下翻转，也不把图像变形到不合理的相机方向。缩放保留宽高比，填充标签为 ignore。

每个 epoch 在验证集评估，联合训练按两个数据集 mIoU 的平均值选择权重，同时保留完整混淆矩阵、各类别像素数、铺装精度/召回。缺失类别的指标单独标明，不把只有五类真值的 CamVid 结果说成已验证水面识别。

固定计划先只运行 seed=42。若要比较其他种子或超参数，继续仅用训练/验证集，完成选择后再固定最终测试。小规模训练是程序检查，5 轮 CamVid 是预热；两者都不能作为装车自动驾驶的最终模型。

## 直接运行

所有命令从项目目录运行，启动器会加载已有 PyTorch/CUDA 环境，模型与数据都使用本地文件。

```bash
cd <项目目录>

# 验证所有真实图像/标注并生成合并清单（完整下载后执行）
/usr/bin/python3 tools/prepare_road_datasets.py --datasets camvid rugd

# 检查全部参数可训练的联合训练链路
./start_road_training.sh --config config/road-train-joint-smoke.yaml

# A 阶段预热；若报告显示已经完成，无需重跑
./start_road_training.sh --config config/road-train-warmup.yaml

# B 阶段联合训练
./start_road_training.sh --config config/road-train-joint.yaml

# 中断后从最后一个完整 epoch 恢复
./start_road_training.sh --config config/road-train-joint.yaml \
  --resume work/road_training/runs/road-joint-v1/last
```

脚本拒绝意外覆盖已完成的运行目录。`best/model` 是验证集选出的模型；`last/model` 与 `last/trainer.pt` 用于继续训练。恢复从最后完整 epoch 开始，中途尚未保存的 epoch 重做。

```bash
source ./activate_project.sh

# 离线看预测效果
python tools/preview_road_model.py \
  --model work/road_training/runs/road-joint-v1/best/model \
  --manifest datasets/public_road/prepared/camvid-rugd-val.jsonl \
  --output work/road_training/preview-joint

# 权重选定后，一次完整独立测试
python tools/evaluate_road_model.py \
  --model work/road_training/runs/road-joint-v1/best/model \
  --manifest datasets/public_road/prepared/camvid-rugd-test.jsonl \
  --output work/road_training/joint-test.json

# ONNX 导出并比对数值
python tools/export_road_model.py \
  --model work/road_training/runs/road-joint-v1/best/model \
  --output work/models/road/road-joint-v1.onnx
```

完整联合训练尚需较多计算，正式训练时关闭 RViz 可减少图形负担。训练时间以本机阶段计时估算；Orin 的 8GB 为共享内存，不按独立显存理解。下载归档、解压数据、处理标签和权重预计占十几 GB，训练不按自动扩张的实验数量无限运行。

## 部署如何衔接

导出固定 384×512 输入的 ONNX，检查 logits 与 PyTorch 一致，然后在该 Orin 上构建 TensorRT FP16。模型预处理必须与训练一致，输出还原时去除 letterbox 填充。降低分辨率或使用 INT8 都需要重新评估精度。

模型先作为只读 ROS 节点订阅 `/camera/color/image_raw`，发布语义掩码和叠加图，供 RViz 查看。随后利用相机内外参把路面区域与深度点投到同一坐标系，生成候选区域及障碍物层。RGB 与深度的分辨率和光学坐标不同，不允许用相同像素下标直接拼接。

相机尚未固定，因此车身外参标定和最终地面过滤留待装车后。路径规划还需要车辆尺寸、轴距、转向限制、里程计等数据。模型分割成绩和当前 30 Hz 图像流不能替代整车速度、制动和避障验收。

## 来源与使用条件

- [SegFormer 官方代码](https://github.com/NVlabs/SegFormer)，研究/评估用途许可；模型文件来自 [NVIDIA 模型仓库](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)，修订 `21b3847fae21ddee674abd31129307b6a1235bd9`。原始权重与官方 LFS SHA256 比对，派生 safetensors 另记录哈希。
- [CamVid/SegNet 教程仓库](https://github.com/alexgkendall/SegNet-Tutorial)，仓库注明个人与研究用途 CC BY-NC 4.0；本地保存来源和原说明。
- [RUGD GitHub 下载说明](https://github.com/dataset-ninja/rugd/blob/main/DOWNLOAD.md)、[原作者网站](http://rugd.vision/)。公开下载不等于已确认任意商用或再分发许可，保留作者资料用于本地研究。
- [YOLO11 官方模型](https://github.com/ultralytics/assets/releases/tag/v8.3.0)，沿用 Ultralytics 的模型许可。已有碗模型不覆盖、不重新训练。

来源快照、许可证资料、归档与权重 SHA256 见 `work/road_training/sources/` 和最终下载清单。归档 CRC 和本地 SHA256 是完整性记录；除已明确提供官方 SHA256 的权重外，不宣称取得了作者发布的校验和。
