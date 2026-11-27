# M2 Pro 机械臂抓碗：MuJoCo 原生离线实验包

面向 M2 Pro / 16GB / macOS 27；包含 arm64 Python 3.12、完整 wheel 依赖、用户 STEP 转换后的网格、候选关节模型、PPO 训练/续训/评估和原生 GUI。无需 Docker、CUDA 或云端环境。

**当前是可运行的候选模型实验包，尚未完成真实机械臂关节标定、MG23 动态联动和泡沫参数验证。** CAD 几何通过与训练模型适用于实机是两件不同的事。本包不连接实体机械臂，也不输出舵机脉宽指令。

独立基线 200 次成功 114 次（57%），尚未达到 95% 目标；短训策略仅做链路检查。详细记录见 [验证范围](validation/验证范围.md)。

解压后在此目录的终端执行：

```bash
bash setup.sh
bash run.sh inspect
bash run.sh gui --cad-only --experimental-model
bash run.sh baseline --episodes 20
```

确认模型限制后，可先跑短训，再续训：

```bash
bash run.sh train --experimental-model --smoke --steps 1024 --output outputs/smoke
bash run.sh train --experimental-model --steps 100000 --output outputs/run01
bash run.sh train --experimental-model --resume outputs/run01/final.zip --steps 100000 --output outputs/run01
bash run.sh evaluate --experimental-model --policy outputs/run01/final.zip --episodes 200 --output outputs/eval01
bash run.sh gui --experimental-model --policy outputs/run01/final.zip
```

详细模型限制、奖励、停车变化、指标、调参顺序和 Mac 操作见 **[Mac调参与仿真指南.md](Mac调参与仿真指南.md)**。本机验证记录在 `validation/`。Mac 二进制经过架构、哈希及依赖检查；macOS GUI 和运行速度须在实际 Mac 上验收，不能把 Linux 检查写成 Mac 已通过。
