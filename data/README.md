# 数据与标注索引

2026-11-02 已补训练/导出实现和既有划分说明，完整数据不放入普通 Git。

| 数据 | 已有说明与实现 | 数量/边界 |
| --- | --- | --- |
| 自采 target_bowl | [标注指南](../software/jetson/docs/annotation-guide.md)、[训练](../software/jetson/docs/target-bowl-training.md)、`tools/export_bowl_dataset.py` | 历史团队报告 500 多张；完整标注与确切划分清单未随本次导入 |
| CamVid + RUGD | [道路训练方案](../software/jetson/docs/road-training-plan.md)、`tools/prepare_road_datasets.py`、`config/road-labels.json` | 本机记录训练 5,185、验证 857、测试 2,095；按完整视频/场景隔离 |
| Gemini MAX RGB-D | [小车接入说明](../software/jetson/cart/README.md)、`cart/capture.py` 与 `cart/rgbd_audit.py` | 2026-11-01 台架记录含 321 RGB / 320 深度及另一段带感知输出的采集；非现场/装车验收 |

完整公开数据和训练交付包仍位于原工作工程/离线交付存储；原始 RGB-D/bag 位于 `software/jetson/` 所对应原工程的 `work/cart_integration/recordings/`。这些本机相对路径是恢复位置，不是仓库下载链接。

相邻视频帧不能跨训练与测试。原有报告中的随机 8:1:1 碗数据划分属于历史记录，需要按采集会话复核；当前有导出检查工具，不据此补写缺失的 mAP。

公开数据来源和使用条件保留在道路训练说明中；未把原始数据或训练权重随源码再次发布。采集、标注和原始文件索引见 [results/](../results/README.md)。
