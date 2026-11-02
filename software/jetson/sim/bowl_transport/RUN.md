# 运行说明：抓碗方案与本机审计 v0.2

此包包含方案、已核对的 CAD、可运行的流程预演、参数化 USD 场景及云端安装脚本。
**尚未包含可训练的自定义机械臂环境、机器人关节 USD 或训练完成的抓碗策略。**
先阅读 `docs/bowl-transport-review-v0.2.md`，它修正了 v0.1 中默认提前准备的决策。本版默认关闭行驶中展开，先验收停车基线。
**当前本机检查通过，但训练租赁准入仍为 false，建议暂不租长时训练服务器。** 后面的云端步骤保留供条件齐备后使用。

## 1. 当前 Jetson 或普通电脑：查看流程预演

在项目/解压包根目录执行，Python 3.10+，不需要第三方包：

```bash
python3 sim/bowl_transport/preflight.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory sim/bowl_transport/output
```

浏览器打开 `http://127.0.0.1:8765/report.html`，或直接打开该 HTML 文件。
本机项目目录为 `/home/nvidia/Documents/ChatGPT/design小组作业`。

输出：`output/report.html` 是总报告；`preflight.json` 保存检查结果、阻塞项和代码哈希；`test-results.txt` 保存故障测试日志。奖励、简化模型、执行器延迟另有 JSON。`timeline.html` 仅展示合成成功信号驱动的动作顺序，不代表实机用时或抓取成功。

整个入口只使用 Python 标准库，不访问相机、串口或机器人。它执行本机测试，不启动 Isaac 或强化学习。当前数值模型只是未经标定的小幅单模态响应，不能计算真实洒水量；满水工况会涉及模型未包含的溢流。

自动化调用可使用：

```bash
python3 sim/bowl_transport/preflight.py --require-training-ready
```

退出码：`0` 对应要求范围内通过；`1` 表示本机检查失败；`3` 表示本机通过但真实训练门槛未满足。本版该选项预期返回 `3`。准入项不能通过直接编辑布尔值解决，需补充可追溯验证记录后更新验收逻辑。

运行逻辑测试：

```bash
python3 -m unittest discover -s tests -p 'test_bowl_transport*.py' -v
```

## 2. 租赁机器

优先 Ubuntu 24.04 x86_64 + RTX 4090 24GB + 64GB RAM + 300GB 持久 SSD。
服务需支持 GPU 图形/Vulkan、SSH；可视化需要远程桌面或正确配置的 Isaac 显示方案。
先按小时验收环境，保存模型资产和 checkpoint 到持久磁盘。

把准备包上传并解压，例如放到 `/home/ubuntu/bowl-transport-v0.2`。
此后命令中的 `task_project` 是你实际解压目录；在包根目录用 `pwd` 取得。

## 3. 在云端安装固定环境

uv 安装方法见 https://docs.astral.sh/uv/getting-started/installation/ 。也可先建独立工具环境：

```bash
python3 -m venv "$HOME/uv-tools"
"$HOME/uv-tools/bin/pip" install uv
export PATH="$HOME/uv-tools/bin:$PATH"
```

若镜像没有 venv/git，先由镜像管理员安装对应系统包。GPU 驱动先用 Isaac Sim Compatibility Checker 核对，不在 Jetson 上执行这些步骤。

在准备包根目录执行：

```bash
task_project="$(pwd)"
bash sim/bowl_transport/cloud_setup.sh "$HOME/IsaacLab-bowl"
```

脚本检出已固定的 Lab 提交并通过 uv 安装该提交声明的 Isaac Sim 6.2.0.0 与 RSL-RL。首次安装/启动需下载依赖与缓存，允许访问 NVIDIA 资产服务。软件许可按实际安装器提示处理。

此脚本已做 shell 语法检查，**尚未在 RTX 云端执行**；版本来源和当前验证范围见 `versions.json`、`validation.json`。

## 4. 云端打开场景

继续在同一终端执行，保留上面的 `task_project` 变量：

```bash
cd "$HOME/IsaacLab-bowl"
uv run --extra isaacsim --extra rsl-rl python \
  "$task_project/sim/bowl_transport/view_in_isaac.py" --headless --frames 240
```

这是有界的无窗口场景加载与更新检查。需要查看画面时，在有可用显示会话的终端去掉 `--headless --frames 240`：

```bash
uv run --extra isaacsim --extra rsl-rl python \
  "$task_project/sim/bowl_transport/view_in_isaac.py"
```

预期看到两个橙色平台、一个参数化黑碗与蓝色视觉水面；没有可运动的自定义机器人。这是资产起始场景，不是抓取演示。场景文件位于 `output/bowl_platform_fixture.usda`，也可在 Isaac Sim 中用 File → Open 打开。

修改 `scenario.json` 后，追加 `--rebuild`，会在 Isaac runtime 内重新生成场景：

```bash
uv run --extra isaacsim --extra rsl-rl python \
  "$task_project/sim/bowl_transport/view_in_isaac.py" --rebuild --headless --frames 240
```

## 5. 检查 Isaac Lab 的官方训练链路

下面训练的是官方 Cartpole，仅用于确认 GPU、Lab 与训练依赖能工作，**不是抓碗任务**：

```bash
uv run --extra isaacsim --extra rsl-rl isaaclab train \
  --rl_library rsl_rl --task Isaac-Cartpole-Direct \
  --num_envs 16 --max_iterations 2 --visualizer none physics=isaacsim_physx
```

该命令依据对应官方安装与 CLI 文档整理，尚待目标云端验证。若命令报参数错误，先用同一环境 `isaaclab train --help` 核对固定提交的 CLI，不改用不匹配版本示例。完成后记录 GPU、驱动、Sim/Lab 版本、启动日志和 checkpoint 位置。

## 6. 开始本项目物理抓取训练之前

必须先把 `assets/cad/chassis-v3` 的机械臂/夹爪建立为验证过的 articulation，并完成关节轴、限位、驱动、耦合、质量惯量、工具中心与碰撞校验。记录空碗质量；流体版使用空碗质量加水质量，总和应为 0.65 kg。

顺序为：单关节检查 → 无水接触抓放 → 650 g 等效负载 → 少量满水物理验证 → 16 环境批量检查 → 接入监督器和有界策略训练 → 留出场景评估 → 实机对应验证。

完整设计见根目录 `docs/bowl-transport-simulation-20261023.md`。当前包没有虚构 `train_bowl.py` 入口；完成真实资产与任务实现后再提供项目训练命令。
