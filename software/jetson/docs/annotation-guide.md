# 明天的碗图像标注

工具：Labelme 7.8.1，安装在独立 `.venv-labelme` 环境，与 Jetson 训练环境分开。

已验证桌面启动、递归读取图片、中文界面、负样本 JSON 保存，以及导出程序的 6 项数据检查。依赖版本记录在 `requirements-labelme.lock.txt`，安装依赖检查通过。当前没有启动训练。

## 打开

双击桌面 **碗轮廓标注**。也可以在项目目录执行：

```bash
./start_annotation.sh
```

打开其他图片文件夹：

```bash
./start_annotation.sh "/你的图片文件夹"
```

工具已固定类别 `target_bowl`，自动保存为与图片同名的 Labelme JSON；原图保留，JSON 不嵌入图片数据。

## 标注动作

1. 在文件列表选择图片。使用“创建多边形”，快捷键 `Ctrl+N`。
2. 沿碗可见的外轮廓点击，双击闭合，类别选择 `target_bowl`。
3. 不包含手、桌面和背景；不补出被遮挡的轮廓。同一图片中所有目标碗都要标。
4. `Ctrl+S` 保存，`D` 下一张，`A` 上一张。自动保存已开启，结束前仍建议按一次保存。
5. 对于其他碗、杯盘和空桌面，确认没有目标碗后，勾选图像标志 **negative**，再保存。不要把“还没标”当成“没有目标”。

实例分割的定义：一只目标碗对应一个实例掩膜，标注属于这只碗的可见像素区域。当前 `target_bowl` 定义为整只碗，不是仅碗口。可见的碗内壁、碗底和碗沿都属于碗；手、食物和背景不属于碗。不要在同一个类别下混用“整碗轮廓”和“仅碗口轮廓”。

多边形内部会被填成掩膜，因此只沿最外边缘画一圈，可能把内部遮挡物也包含进去。当前导出器按一个多边形表示一个实例，不支持带孔掩膜或同一实例的多个分离区域。第一轮优先使用空碗、无遮挡图片；遇到形成孔洞或把碗分成几块的遮挡，先单独留待复核，不要把遮挡物圈进碗，也不要把同一只碗拆成多个实例强行导出。

先用多边形工具。AI 功能可能下载额外模型；自动轮廓也必须人工检查。当前没有下载或预加载标注辅助模型。

## 放置图片与划分

```text
datasets/target_bowl/source/
  train/拍摄批次A/图片.png + 图片.json
  val/拍摄批次B/图片.png + 图片.json
  test/拍摄批次C/图片.png + 图片.json
```

建议按独立拍摄批次分成约 80%/10%/10%，同一段视频的相邻帧放在同一个划分中。每个划分需要目标碗正样本，也应覆盖其他碗/背景负样本。以完整批次文件夹为单位移动时，原图与同名 JSON 一起移动。

已放入一张今天的目标碗参考图，位于 `train/session-reference/`，仅作为打开工具的起点，尚未标注。其他采集图片请复制到相应批次目录，工具不会自动移动或删除你在其他位置的照片。

## 导出为 YOLO 实例分割

标注完成后先检查：

```bash
source ./activate_project.sh
python tools/export_bowl_dataset.py --check-only
```

检查通过再导出：

```bash
python tools/export_bowl_dataset.py
```

默认输出 `datasets/target_bowl/yolo/data.yaml`、`images/{train,val,test}`、`labels/{train,val,test}` 和原图对应清单。导出不修改源文件；若输出已存在，使用 `--output datasets/target_bowl/yolo-v2` 保存新版本。

程序检查标签名、图像尺寸、坐标、自交轮廓、未标注图、负样本确认和跨划分重复图片。当前数据不足或标注尚未完成时，检查报“尚不能导出”是正常提醒，不会生成假标签。

## 训练

明天检查数据后，再开始训练第一版 `YOLO11n-seg`。模型在 Jetson 上运行；标注工具不自动启动训练。训练入口以导出的 `data.yaml` 为准。

```bash
source ./activate_project.sh
yolo segment train model=work/models/yolo11n-seg.pt \
  data=datasets/target_bowl/yolo/data.yaml \
  imgsz=320 epochs=80 batch=2 device=0 workers=2 \
  project=work/bowl-training name=target-bowl-v1
```

这里的轮数和批量是初始配置，不保证识别精度。先保留独立测试集，训练后再检查误检、漏检与轮廓质量。

参考：[Labelme 官方项目](https://github.com/wkentaro/labelme)、[YOLO 分割标签格式](https://docs.ultralytics.com/datasets/segment/)。
