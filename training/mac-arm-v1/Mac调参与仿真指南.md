# MacBook 机械臂抓碗训练与调参指南

版本：v0.1，2026-11-17。目标电脑 M2 Pro、16GB、macOS 27。首轮目标是停稳后的抓取和放回成功率，底盘行驶、避障和满水防洒水不进入本轮训练。

## 1. 交付内容和当前完成度

这是原生 arm64 离线环境包，压缩包包含 Python 运行时和全部 pip wheels，不要求先安装 Python、Homebrew、Docker、CUDA 或 Rosetta。它不是 Docker 镜像；原生运行便于使用 MuJoCo GUI，避开额外虚拟机与图形转发。

已经完成：

- 读取用户 `机械臂带爪子.STEP`，用 OpenCascade 转换 41 个独立几何部件、保留 171 个叶子装配实例和各自变换，41 个部件的 B-rep 有效性检查均通过。网格单位已从毫米转换为米。
- 从 CAD 舵机轴柱面推导候选关节轴；与相机安装信息一致，相机安装参考点挂在 004 末端旋转之前。
- 目标依赖随机停车平移/朝向转换到臂基座坐标，重新求取抓放轨迹，不回放固定关节动作组。
- 自由刚体碗、接触/抬升/承托评价、候选四指夹持接触、PPO、断点保存/恢复、独立评估、原生 GUI。

必须明确的近似：

- STEP 不包含已验证的运动关节模型。关节轴正负、零位、限位、惯量、力矩、刚体分组和舵机延迟尚未实测。电机壳/舵盘和相机按结构指定分组，其余部分采用候选分组，GUI 中需要逐关节核对。
- **MG23 目前是四个刚性指部包络、单控制量、径向联动的近似模型。** 保留真实指部显示网格和泡沫块位置，但没有重建 MG23 全部连杆闭环、齿条传动及指节被动自适应。不能把当前训练权重称为“真实 MG23 已验证策略”。
- 硬夹指碰撞使用网格凸包。内凹和细节接触可能与真实夹指不同；泡沫接触用可配置的等效软接触，并不等于已识别的泡沫材料模型。CAD 的缓冲垫厚度也不等于实物泡沫实测厚度。
- 黑色塑料碗按照片/示意的近似旋转轮廓建模，口径约 156 mm、高约 60 mm；满载总质量 0.65 kg 包含碗和水。空碗/水质量未拆分，水作为等效刚体载荷，没有液面、晃动或洒水量输出。
- 当前策略观测中的目标位置由带噪声和延迟的坐标估计器模拟，**尚未接入 RGB 相机、真实手眼标定或已有分割模型**。真实视觉接入后必须提供同一基座坐标系、同一单位、匹配图像时刻的目标估计与有效性。

完整检查使用 20 Hz 控制采样点的接触证据；本版不声称对相邻采样点之间的全部碰撞峰值做了连续认证。需要更严格碰撞审核时，应增加物理子步事件记录并重新评估。提供的 STEP 转换脚本用于追溯生成过程，常规训练只读取已转换网格；可选 OpenCascade 工具依赖不包含在 Mac 训练运行时内。

为避免直接把实验权重用于实机，正式学习入口要求明确添加 `--experimental-model`。这只是表明你要实验该候选模型，不会把 `hardware_ready` 改为 true。

## 2. 解压和离线安装

把压缩包复制到 Mac 的普通本地目录，先解压。终端进入**解压后的包根目录**（能看到 setup.sh、run.sh、payload、assets）：

```bash
cd "/你的实际解压目录/mac-arm-v1"
bash setup.sh
```

路径可以含中文和空格，务必加引号；上面的路径是占位示例，请替换为实际路径，也可以在终端输入 `cd ` 后将文件夹拖入终端。

安装脚本先校验 SHA256，再解压 Mac arm64 Python 3.12，建立 `.venv`，用 `--no-index --require-hashes` 从自带 wheels 安装，最后执行 `pip check` 和模型检查。首次安装需要等待文件解压，整个过程不需要外网。**不要复制 Jetson/Linux 的 `.venv` 到 Mac。**

要求运行在原生 arm64 macOS；若终端显示 `x86_64`，请使用原生终端。SciPy wheel 要求 macOS 14 或更新；目标 macOS 27 满足版本门槛，但仍需要实际启动验收。若系统安全提示阻止打开下载内容，按 macOS 正常的来源验证与打开流程处理；本包不自动移除隔离属性或关闭系统保护。

安装后输出文件在 `outputs/`，原始网格和模型在 `assets/`。不要移动或改名已经安装的目录；需要换位置时先移动原始解压包，再运行 setup.sh 重建环境。

## 3. 先查看真实 CAD 和检查模型

```bash
bash run.sh inspect
bash run.sh gui --cad-only --experimental-model
```

GUI 使用官方 `mjpython` 启动，鼠标拖动调整视角；空格暂停/继续，R 回到起点，Q 退出。CAD-only 展示导出姿态。它还不是碰撞抓取演示。

MuJoCo 左侧关节面板可以用来查看轴线运动。重点检查：000 底座、001 肩、002 肘、003 腕俯仰、004 末端旋转的方向与刚体归属，尤其是加长部件和连杆；相机不能随 004 转动。`camera_mount_reference` 只是 CAD 安装参考点，不能作为标定好的光学坐标系。

候选抓放回放：

```bash
bash run.sh gui --experimental-model
```

脚本先计算一回合，打印成功或失败原因，再回放。回放窗口出现不等于回合通过；失败轨迹也保留供排查。界面显示的夹爪径向闭合属于上述近似，不能用它证明真实 MG23 传动正确。

## 4. 停车位置和朝向不固定如何处理

在 `config/train.yaml` 的 `parking` 中配置实验范围。默认前后/左右各 ±2.5 cm、朝向 ±15°，**这不是用户停车精度的实测值，只是首轮实验设置**。没有对“任意停车位置”作保证。

每回合先采样停车姿态 `(x, y, yaw)`，将平台、碗的世界位置变换到机械臂基座坐标系；根据带误差的目标估计重算预抓取、接近、抬升和放置路径。末端平移修正交给策略，整体位置变化由重规划承担，不要求小范围残差覆盖所有停车误差。不可达目标终止为 `unreachable_reposition_required`，需要上层停车/微调车位，不能计为成功。

扩大范围时，先生成可达网格：

```bash
bash run.sh workspace --experimental-model --output outputs/workspace
```

它扫描前后、左右 ±8 cm、朝向 ±45°的 125 个候选停车姿态，输出 `workspace.json`。该结果只是候选关节/限位模型的运动学筛查，不是完整轨迹无碰撞证明。进一步扩大训练范围应在检查结果和现场停车记录后进行。

当前每回合在一次停稳状态完成局部“抓起—保持—放回”技能。实际小车跑一圈返回后，即使仍是同一个平台、同一个世界放碗点，新的停车姿态也要重新估计平台目标在当前臂基座下的位置。**不能沿用出发时的关节角或旧的相机外参变换结果。** 当前包没有仿真中间整圈行驶，也未实现自动调整车位的底盘控制。

## 5. 第一次运行顺序

先检查本包测试和确定性基线：

```bash
.venv/bin/python -m unittest discover -s tests -v
bash run.sh baseline --episodes 20 --output outputs/baseline20
```

随后做 1024 步短训，检查并行环境、优化器和保存：

```bash
bash run.sh train --experimental-model --smoke --steps 1024 --envs 2 --output outputs/smoke
```

`smoke` 只证明训练程序可以运行，不是抓碗学会了。能看到 `final.zip`、`run.json`、每环境 CSV、TensorBoard 日志和 checkpoint。不要把示例权重命名为已验收的最佳策略。

正式实验使用新目录，先 10 万步观察：

```bash
bash run.sh train --experimental-model --steps 100000 --envs 2 --output outputs/run01
```

确认内存与速度后再考虑增加并行数；16GB 起步 2 个环境，最多建议先试 4 个并测吞吐。训练使用 CPU，关闭 GUI；Apple GPU/MPS 不是本版的默认路径。终端暂时没有输出时看 TensorBoard/CSV 和进程状态，不立即删日志重装环境。

查看曲线：

```bash
.venv/bin/tensorboard --logdir outputs/run01/tensorboard --host 127.0.0.1 --port 6006
```

浏览器打开终端显示的本机地址。总奖励、`task/success`、`task/reachable`、`task/place_error_m` 共同查看；训练期间的 episodic 指标是诊断，最终结论以独立评估为准。自动 `best_model.zip` 按验证平均回报选择，仍须与 `final.zip` 分别比较成功率，不能把它自动等同于最可靠模型。

## 6. 续训、退出和独立评估

Ctrl+C 会保存当前 `final.zip`。继续追加训练步数：

```bash
bash run.sh train --experimental-model --resume outputs/run01/final.zip --steps 100000 --output outputs/run01
```

`--steps` 是本次追加步数。保留 checkpoint 所在目录的 `run.json`，否则无法校验模型/配置一致性。改变配置或几何后建立新实验，不把不兼容的旧 checkpoint 当作直接续训。保存会恢复网络、优化器等 PPO 状态，但环境随机数和 rollout 不保证逐比特续接；记录种子与配置用于可复核实验。

固定独立种子，比较同一配置下控制基线与学习策略：

```bash
bash run.sh baseline --seed 10000 --episodes 200 --output outputs/baseline200
bash run.sh evaluate --experimental-model --policy outputs/run01/final.zip --seed 10000 --episodes 200 --output outputs/policy200
bash run.sh gui --experimental-model --policy outputs/run01/final.zip
```

输出 `evaluation.json`，同时报告全部 200 次成功率、可达比例、可达样本成功率、失败原因、95% Wilson 区间和每回合停车姿态。不可达回合留在总体分母中，不能只挑容易的位置来宣称总体成功率。

默认训练种子 42，自动验证种子 1000000，最终比较从 10000 开始。模型选择可以看自动验证；不要用最终测试不断选择参数再声称它是全新留出测试。调参用完一组测试后，再建立独立新种子集复核。

首轮暂定目标：至少 200 次独立测试、全流程成功率 ≥95%；碗底离台 ≥30 mm 并连续保持 2 s，放回平面误差 ≤10 mm，台面承托、松爪脱离、稳定放置，同时没有碰撞、掉落或越限。`passes_provisional_simulation_goal` 只代表上述候选仿真目标，不代表实机或满水防洒水验收。

## 7. 奖励函数具体实现

见 `arm_training/rewards.py`。每个物理控制周期 50 ms：

| 成分 | 首版权重与条件 |
| --- | --- |
| 接近整形 | `5 × (exp(-λdt) Φ(s') - Φ(s))`，`Φ=-clip(位置误差/0.05, 0, 1)`，终态势设 0 |
| 试抬 | 双侧对向接触存在、碗底离台超过 10 mm 后一次性 +10 |
| 抬升保持 | 进入保持阶段、碗底离台 ≥30 mm、倾角合格、对向接触持续 2 s 后一次性 +20 |
| 完成 | 放回误差 ≤10 mm、台面承托、夹指脱离、速度稳定、已经验证保持后一次性 +100 |
| 失败 | 不可达、输入无效、非预期碰撞、越限、掉落、保持失败、承托缺失、最终放置失败或任务超时，终止并 -200 |
| 耗时 | 每秒 -0.2 |
| 倾角 | 每秒 `-0.5 × min((倾角/0.15)^2,4)` |
| 残差幅度 | 每秒 `-0.02 × sum(a²)`，a 在 [-1,1] |
| 动作变化率 | 每秒 `-0.0005 × sum(min(((a-a_prev)/dt)^2,400))` |

`λ≈0.201006717/s` 对应 50 ms 时 `gamma=0.99`。如果修改控制周期，同时调整 PPO gamma 与折扣率，保持时间定义一致。阶段位置势在本实现中随运动参考变化，不能无条件宣称具有严格策略不变性；成功率和故障对照仍是最终检验。

不按夹力大小加分，不用仅闭爪计时冒充抓住，不把倾斜代理当作洒水克数。仿真接触力只供监督器验证阶段和统计奖励，策略观测不包含精确接触力；真实部署时仍需设计相应的夹持、试抬和承托证据，不能凭空假设有力传感器。

## 8. 调参顺序

每次复制 `config/train.yaml` 为新配置，记录实验名，例如 `config/experiment01.yaml`。命令把全局 `--config` 放在子命令前：

```bash
bash run.sh --config config/experiment01.yaml baseline --episodes 20 --output outputs/experiment01-baseline
bash run.sh --config config/experiment01.yaml train --experimental-model --steps 100000 --output outputs/experiment01
```

建议顺序：

1. **关节与几何**：核对实际刚体分组、轴线、方向、零位、限位和夹爪联动。修正 `assets/candidate-model.json` 需由清楚几何关系的人操作，保留原文件。现在的自动分组只是起点。
2. **接触高度/闭爪行程**：先用不带学习的候选搜索，看哪个位置能持续夹持。`bash run.sh search --experimental-model` 搜索 3 个接触高度和 4 个行程，输出结果但不自动覆盖配置。不要为获得成功而任意提高摩擦或无限夹力。
3. **执行器**：记录真实位置指令和反馈，匹配速度、滞后、死区/回差；当前只实现延迟和限速，没有完整回差模型。控制 20 Hz 不等于实物跟踪带宽 20 Hz。
4. **感知误差**：通过标定与真实数据设定误差/延迟，再扩随机范围。当前默认 3 mm 初始估计误差和 1 mm 观测噪声只是候选值。
5. **停车范围**：先查看可达网格和失败位置，再逐步扩范围；不可达需求交回停车规划。实际允许范围要从现场数据定义，不能把默认 ±2.5 cm 当成车停得这么准。
6. **策略**：前三项合理后再改学习率、步数和动作范围。当前网络 64×64、学习率 3e-4，动作是三维末端位置残差，最大 ±10 mm、变化率限 30 mm/s。夹爪开合由阶段控制，策略不决定松爪准入，也不输出力矩。

常见失败对应项：`grasp_contact_missing` 查接触高度、夹爪开口/行程和观测偏差；`lift_hold_not_verified` 查滑脱/倾角/负载和泡沫；`unexpected_collision` 查实物分组、预抓取位置、台面与路径；`joint_tracking_limit` 查动作速度与延迟，不直接放宽阈值；`unreachable_reposition_required` 查停车位置、臂高、关节限位和 IK。参数变化后用相同条件对照基线，再用新的留出条件验证。

## 9. 从 Mac 到 Nano 的下一步

Mac 上训练后的策略是 SB3 `zip`，可在 Nano 的匹配推理栈上加载；当前包不自动部署。先补 `config/hardware-calibration.template.json` 中的关节脉宽/角度、MG23 开口与联动、相机内外参、舵机动态、泡沫和承托证据。当前确认相机不随 004 旋转，但照片不能确定毫米级外参。

实机对照分阶段：目标坐标离线检查、空载低速路径、空碗、等效总载荷、再满水。基线和策略采用相同硬限位、速率限制和阶段准入。需要独立记录真实成功率、滑移、承托、碰撞、任务时长与失水量。只有仿真合格没有这些证据时，`hardware_ready` 保持 false。

## 10. 运行失败时保留什么

保留终端完整错误、`outputs/inspection.json`、对应实验的 `run.json`、`config-used.json`、`final.zip`、`evaluation.json`。这些足够区分安装、几何、策略和评价问题。不要发送整个 `.venv` 或重下大环境；本包已包含离线依赖。若 macOS 原生 GUI 报线程错误，确认使用 `bash run.sh gui ...`，脚本会调用 `mjpython`。

## 11. 借鉴项目和许可证

- [MuJoCo Menagerie / Robotiq 2F85](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotiq_2f85)：参考闭环夹爪、联动与接触建模思路，未用其几何替代 MG23。
- [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics)：参考随机目标、相对目标观测与独立成功判定；没有复制 Fetch 的机器人和观测真值假设。
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)：直接使用其 MIT 许可 PPO、并行环境、保存和恢复接口。
- [robosuite](https://github.com/ARISE-Initiative/robosuite)：参考模块化抓放任务组织，未引入整套框架依赖。

实际查询的提交/许可证在 `docs/upstream-references.json`。用户 CAD 和照片属于用户资料，不因打包而套用上述项目许可证；本包未上传这些资产。Python/wheel 各自许可证保留在运行时与安装包内。

官方接口参考：[MuJoCo 3.3.7 Python/GUI](https://mujoco.readthedocs.io/en/3.3.7/python.html)、[MuJoCo 接触模型](https://mujoco.readthedocs.io/en/3.3.7/modeling.html)、[SB3 PPO](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html)。
