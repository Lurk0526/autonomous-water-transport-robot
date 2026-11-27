# Mac 机械臂实验包 v0.2：先修控制和模型，再决定是否训练

适用：M2 Pro / 16GB / 原生 arm64 macOS。依赖保持 Python 3.12.16、MuJoCo 3.3.7、PyTorch 2.8.0、SB3 2.7.0；修复 TensorBoard 所需的 setuptools 为 80.9.0。该包为原生离线环境，不需要 Docker、CUDA 或 NVIDIA 仿真平台。

## 1. 这次调整的判断

你的 Mac 首轮报告中，固定的 200 个新场景上，规则基线为 122/200（61.0%），最佳策略为 123/200（61.5%），最终策略为 108/200（54.0%）。最佳策略仅独有成功 13 次，基线仅独有成功 12 次，配对检验 p=1.0，没有可靠学习增益。

你补充确认：**实机夹爪可以夹起满载 650g 的碗并稳定保持至少 2 秒。** 因此先修仿真控制、夹爪运动和接触建模。不能用仿真失败反推实机夹力不足，也不应直接增加训练步数。

本轮推荐路线是：现有视觉估计碗相对基座的位姿 → 可达性检查 → 协调关节运动与抓放状态机 → 真实动作/接触标定 → 必要时学习小幅残差。停车位置、车身朝向变化需要重新变换目标坐标、重算轨迹；这本身并不要求强化学习。

## 2. v0.2 已修改什么

| 问题 | 修改 | 保留的边界 |
|---|---|---|
| 各关节分别限速会破坏保持水平所需的配合 | 每次更新用同一比例缩放全部关节位移，最大关节速度仍为 0.7rad/s | 关节动力学和限位仍待标定；并非完整避障规划 |
| 固定时钟切换，没稳住也闭爪/下降/松爪 | 每阶段运动时间结束后，等待到位、速度稳定及该阶段的抓持/承托证据；等待超时明确失败 | 接触证据来自仿真；实机传感器接入尚未完成 |
| 50ms 控制末端采样会漏掉短暂碰撞 | 每个 1ms 物理步检查不允许的接触，保留首次零件名、阶段、时间、力和位置 | 0.5N 阈值保留，没有关闭碰撞 |
| 放回时仍可施加 ±10mm，等于成功容差 | 最大残差改为 5mm；降低阶段 2mm，承托/松爪/退开阶段逐步归零 | 成功的 XY 容差仍为 10mm |
| 策略可以移动自己的奖励目标 | 距离塑形改对未加残差的名义轨迹计算 | 完成/失败事件仍为主，不承诺一定收敛 |
| PPO 折扣与奖励折扣分离 | 从 PPO 的 gamma 和控制步长计算塑形折扣 | 改 gamma 必须新建实验 |
| 20 个变化场景按平均奖励选 best | 固定种子 30000–30099 共 100 场景，先按成功数，再按碰撞数、放置误差选策略；单独保存零残差基线 | best_model 只表示本次策略中最好，不代表比规则基线好 |
| evaluation_frequency 没生效、target_kl 没传入 | 两项均接入，评估间隔按并行环境数换算 | 实际更新仍按 PPO rollout 对齐 |
| TensorBoard CLI 缺少 pkg_resources | 离线锁定官方 setuptools 80.9.0，安装时检查 TensorBoard CLI | 上游废弃提示可能存在，但不再缺模块 |

阶段依次为：预稳 → 接近 → 闭合 → 抬起 → 保持 → 下降 → 承托确认 → 松爪 → 退开 → 放置稳定确认。

到位要求：各关节距当前 IK 目标不超过 0.06rad，关节速度不超过 0.12rad/s；连续符合 0.15 秒。阶段确认额外等待最多 2 秒，整任务最多 32 秒。保持仍需碗底离台至少 30mm、倾角不超过 0.2rad、对向接触持续 2 秒。放回仍需 XY 误差不超过 10mm，足够台面承托、夹指脱离、线速度低于 0.02m/s、角速度低于 0.1rad/s、倾角符合要求，连续稳定 0.5 秒。抓持/稳定证据按 20Hz 控制周期检查；碰撞按 1kHz 检查。

**这些成功标准没有放宽。v0.2 的检查更严格，因此不能只拿它的原始成功率与 v0.1 当作完全相同的评价口径。** 同条件仅修改协调限速的消融实验单独保存，见 `validation/调整验证报告.md`。

## 3. 安装：使用单独目录

保留 Mac 上现有 `mac-arm-v1` 及其 outputs、TensorBoard 兼容脚本和实时查看器。把新包解压为相邻的 `mac-arm-v2`，不要覆盖旧目录。

如果使用小更新包，解压后在 `mac-arm-v2-update` 目录执行：

```bash
bash apply_update.sh "/你的实际路径/mac-arm-v1"
```

它先校验并复用旧包内未修改的网格、原始资料和运行时归档，在旧目录旁建立 `mac-arm-v2`。旧训练输出和环境不复制、不修改。旧资产缺失或哈希不同会停止，改用完整离线包。已有 v2 目录时不会覆盖，可加 `--destination "/另一个新路径/mac-arm-v2"`。

在生成或解压的 **mac-arm-v2 目录** 打开终端：

```bash
bash setup.sh
bash run.sh inspect
bash run.sh gui --cad-only --experimental-model
bash run.sh baseline --episodes 20 --seed 30000 --output outputs/v2-check20
```

安装完全离线，先逐文件校验再装依赖。若提示架构错误，使用原生终端，关闭 Rosetta。GUI 随 MuJoCo Python 包提供；`run.sh gui` 在 macOS 自动用 mjpython 启动。空格暂停、R 重放、Q 退出。GUI 回放单个种子的独立仿真，不等于训练进程或正式评价。

v0.2 观测改为 45 维，控制、奖励、结束条件也变化。**不能续训或加载旧 v0.1 策略**；旧策略在原 v1 目录复核。加载 v2 策略要求旁边有对应 run.json，且模型、配置和源码哈希一致。

## 4. 先做诊断和零残差基线

```bash
bash run.sh diagnose --seed 30000 --output outputs/diagnose-30000
bash run.sh gui --experimental-model --seed 30000
bash run.sh baseline --episodes 100 --seed 30000 --output outputs/dev-baseline100
```

诊断输出：

- `episode.json`：失败原因、阶段、停车位姿、定位误差、执行器延迟、首个碰撞、阶段确认结果。
- `trace.json`：逐控制周期的接触几何名称/力/位置、泡沫与硬质指节的接触区别、碗底净空、倾角、承托力、期望/实际关节、名义目标和残差。
- `replay.json`：本次完整状态，供复核；不输出实机指令。

优先查看首个失败，而不是终态可能掩盖的原因：

| 失败 | 先检查 |
|---|---|
| unexpected_collision | first_collision 中哪根指节/连杆碰到什么；对照轨迹姿态和实际指尖高度 |
| joint_tracking_limit / motion_not_settled | 目标与实际关节、限速、延迟、饱和、IK 连续性；不要先提高误差阈值 |
| grasp_contact_missing | 是否从泡沫垫接触外壁，还是硬指节勾住碗沿；开口和传动映射是否匹配实机 |
| lift_clearance_not_verified / lift_hold_not_verified | 首次抬起是否偏斜、哪一侧先失去接触、碗底是否达到 30mm |
| support_missing_after_lower / support_missing_release_inhibited | 下降目标与碗的实际位置、是否真正得到台面承托；不绕过松爪条件 |
| release_not_verified | 软垫是否粘带或硬指节卡住；台面承托能否保持 |
| final_placement_not_verified | XY 误差、碗的倾角/速度和连续稳定时间 |
| unreachable_reposition_required | 超出候选可达范围，需要调整停车；仍算入总体失败分母 |

## 5. 先补哪些实机数据

不需要再次提供完整 STEP。现在缺的是**运动和标定证据**：

1. 把已经能稳定抓起 650g 碗的舵机脉宽、顺序、动作时间记录下来；至少有“张开、接触、夹稳、抬起、放回”几个点。若没有关节反馈，标明记录的是指令而非实际角度。
2. 同一固定机位对照夹爪全开、中间、夹住碗三个状态；测每个状态的实际开口、夹指姿态及泡沫接触高度。现有三指节按整体平移近似，不能靠调摩擦系数替代真实联动。
3. 关节零位、轴向、实际限位、基座与台面高度、相机内外参。相机固定在 004 之前、不随 004 旋转这一规则继续保持。
4. 泡沫厚度/压缩、干湿表面差别；650g 是碗和水总重，不是水的质量。空碗重量未知仍保留未知。

先在仿真复现那条真实成功动作，再看是否需要重建 MG23 被动指节/闭环约束，或用实测开合曲线建立经验证的等效模型。**当前网格外观来自用户 STEP，但关节分组、惯量、执行器及四指径向包络尚未通过上述标定。**

`task.require_foam_grasp: false` 表示当前包仍统计对向夹指包络接触，允许硬指节；同时记录 foam_bilateral、foam_forces_n。它不是“软垫真实抓稳”的证据。可复制配置设为 true 做诊断，但不要为了数字好看扩大摩擦/力矩或关闭硬指节碰撞。基线达标后，仍需要接触和运动标定才能谈实机迁移。

## 6. 停车位置和角度变化

默认 x/y 各 ±25mm、yaw ±15°只是可配置实验范围，不是你的停车测量。每回合重新采样车位，把固定世界台面/碗位置变换到当前臂基座坐标，重算抓取与放回轨迹。

```bash
bash run.sh workspace --experimental-model --seed 30000 --output outputs/workspace
```

`workspace` 是端点 IK 扫描，范围 ±80mm、±45°，不证明整段路径无碰撞。`baseline` 才实际执行轨迹、检查接触。到现场用几十次停车统计扩展 x/y/yaw 范围；更换范围重新建立零残差基线，不排除不可达失败来美化成功率。

本包模拟单次停车后的抓取、保持和原位放回，没有包含行驶一圈后重新停车、障碍路径、整车悬挂或水晃动。返程必须重新感知放置平台，在新的车体坐标下重算目标。

## 7. 调参实验：有证据再改

先留默认配置作对照。每个实验复制配置，修改一类参数，记录配置、源码、模型哈希。同一批种子比较才有意义。

```bash
mkdir -p experiments
cp config/train.yaml experiments/slower.yaml
# 用编辑器只修改 task.durations_s 的某一个阶段，例如接近从 1.5 改为 2.0
bash run.sh --config experiments/slower.yaml baseline --episodes 100 --seed 30000 --output outputs/dev-slower
bash run.sh compare outputs/dev-baseline100/evaluation.json outputs/dev-slower/evaluation.json --output outputs/compare-slower.json
```

`compare` 检查种子和停车位姿一致；v2 结果还核对定位初始误差、延迟、摩擦等场景参数，输出候选独有成功、基线独有成功、成功率差和精确配对检验。若你有意改变域参数，分开报告敏感性实验，不强行当作完全配对。

需要搜索接触高度和闭合命令时：

```bash
bash run.sh search --experimental-model --episodes 50 --seed 30000 --output outputs/contact-search
```

搜索 25/35/45mm 接触高度和 25/32/38/45mm **径向平移近似命令**，共 12 组；每组 50 场景，保存完整失败记录，不自动应用最优组。这个行程不是已标定的实机开口/舵机脉宽。没有运动标定前，搜索只用于暴露模型问题。选出的配置再在开发集 30000–30099 上确认；反复尝试过的种子不能再称最终盲测。

## 8. 强化学习的启动条件和命令

目前建议先用零残差基线。只有模型能复现实机成功动作、零残差可靠、残余误差确有可观测信息可纠正时，再做 PPO。不要把所有失败交给网络补偿。可用下面短训仅验证安装、保存和评估链路：

```bash
bash run.sh train --experimental-model --smoke --steps 1024 --output outputs/v2-smoke
```

正式实验从头开始，不续训 smoke：

```bash
bash run.sh train --experimental-model --steps 100000 --seed 42 --output outputs/v2-seed42
```

默认 CPU、2 环境、64×64 网络、学习率 1e-4、gamma 0.995、target_kl 0.02。每 10000 个总训练步固定评估 100 个开发场景。实际训练步数按 `n_envs * n_steps` 对齐，日志中评估时间不计为训练步。

`best_model.zip` 按成功数优先选择；`final.zip` 只是最后状态；`selection.json` 明确保留规则基线为推荐控制器，直至新盲测证明策略提升。若 `best_model` 也没有配对提升，停止加步数，回到失败诊断。用 42、43、44 三个独立训练种子确认稳定性，不能只展示最好的那次。

同一配置和源码的续训使用新输出目录，保留原 run.json 和模型：

```bash
bash run.sh train --experimental-model --resume outputs/v2-seed42/final.zip --steps 100000 --seed 42 --output outputs/v2-seed42-resume01
```

源代码、配置、模型变动后必须新建训练；v1 与 v2 不能混用。`source-integrity.json` 记录运行中源码是否保持不变。

开发集策略对照：

```bash
bash run.sh evaluate --experimental-model --policy outputs/v2-seed42/best_model.zip --episodes 100 --seed 30000 --output outputs/dev-policy
bash run.sh compare outputs/dev-baseline100/evaluation.json outputs/dev-policy/evaluation.json --output outputs/dev-comparison.json
```

最终选定配置和策略后，**仅一次**在未参与选择的 50000–50199 上同时评规则基线和策略；本轮本机控制器对照使用 40000–40199，不能再当后续策略盲测：

```bash
bash run.sh baseline --episodes 200 --seed 50000 --output outputs/holdout-baseline
bash run.sh evaluate --experimental-model --policy outputs/v2-seed42/best_model.zip --episodes 200 --seed 50000 --output outputs/holdout-policy
bash run.sh compare outputs/holdout-baseline/evaluation.json outputs/holdout-policy/evaluation.json --output outputs/holdout-comparison.json
```

仿真目标保持 ≥95% / 至少 200 场景，同时报告 Wilson 95% 区间和失败构成。190/200 仅达到点估计目标，不意味着总体成功率的置信下限达到 95%。如果观察了这组结果又继续改模型/参数，它就成为开发数据，需要再换未用场景。

## 9. 奖励和策略可见信息

一次试抬 +10、保持验证 +20、完整放回 +100、任何终止失败 -200，各事件只发一次，失败步不发成功事件。距离塑形系数 5，势函数为 `-clip(名义轨迹距离/0.05, 0, 1)`，采用与 PPO 相同折扣。每秒时间成本 0.2，倾角成本系数 0.5、残差成本 0.02、动作变化率成本 0.0005；具体公式见 `arm_training/rewards.py`。

观测 45 维：关节位置/速度、近似闭合量、带延迟和噪声的碗相对 TCP 位姿、记忆的初始目标、名义轨迹误差、上步动作、当前阶段、观测有效性/年龄、残差状态、关节指令跟踪误差、阶段进度和总时间。精确接触力、真值放置误差、抓持计时器不直接给策略。阶段由仿真监督器推进，仍会间接体现仿真证据；未来接实机要实现相应可测的监督条件。当前目标观测是仿真位置加噪声，并非 RGB/真实视觉闭环，不能据此声称视觉迁移已经通过。

## 10. TensorBoard 和交付范围

```bash
.venv/bin/python -m tensorboard.main --logdir outputs/v2-seed42/tensorboard --host 127.0.0.1 --port 6006
```

浏览器打开 `http://127.0.0.1:6006`。setuptools 固定为 80.9.0，保留 TensorBoard 2.19 所需的 pkg_resources；不需要旧包临时兼容脚本。根据 [setuptools 官方变更记录](https://setuptools.pypa.io/en/stable/history.html#v82-0-0)，该模块自 82.0.0 被移除。SB3 并行回调间隔按环境数调整，参考 [官方回调文档](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)。

本机执行验证为 Linux aarch64 CPU；此次 Mac 二进制及锁文件做离线哈希/依赖校验，Mac 原生执行须由你在设备上复核。旧报告是 v0.1 的 Mac 实测，不可写成 v0.2 的 Mac 验证。测试通过、短训保存成功和高仿真成功率都不等同于 MG23 标定、流体防洒或实机部署通过。
