# 从另一台机器补充代码

远端创建并完成首次推送后，先克隆这个仓库，再复制已有工程。不要在另一台机器另建一个同名空仓库。

```bash
git clone https://github.com/Lurk0526/AstraGO.git
cd AstraGO
git switch -c codex/import-runtime-code
```

目录只是接收位置，不要求重构已经能工作的工程：

| 原有内容 | 建议位置 | 同时保留 |
| --- | --- | --- |
| Jetson 视觉、推理、任务程序 | `software/jetson/` | 入口、配置、依赖与真实运行命令 |
| OpenMV 程序 | `software/openmv/` | 相机设置、脚本和输出协议 |
| ROS 工程 | `software/ros_ws/src/` | 包、launch、参数与消息定义；不上传 build/install/log |
| STM32 工程 | `firmware/stm32/` | Src/Inc、驱动、启动文件、链接脚本、构建文件、`.ioc`；有则保留 Keil/CMake 工程 |
| 模型训练与评估 | `software/training/` | 数据 YAML、参数、划分方式、版本和导出命令 |
| 通信实现 | 放在所属工程中 | 另在 `hardware/electronics/` 记录协议、引脚与单位 |
| 机械臂 URDF、网格、仿真任务 | `simulation/` | 资源相对路径、关节限位、依赖和启动方式 |

每个真实工程附一份简短 README：板卡/系统版本、入口、依赖、构建/运行命令、接线前提、已测试功能和已知问题。版本应抄录实际设备输出，不根据本仓库技术方向猜版本。

## 权重、数据和视频

当前 `.gitignore` 默认排除了模型权重、训练输出、原始数据集、ROS bag 和视频。先在 `models/`、`data/`、`results/` 写索引：文件名、版本、大小、SHA-256、存储位置、对应代码提交和用途；可提交少量获准共享的样例。

如确实要让大文件随仓库管理，先安装并配置 Git LFS，再修改对应忽略规则并提交 `.gitattributes`。本次没有安装或启用 LFS。不要直接用 `git add -f` 绕过忽略规则批量上传。

[GitHub 普通 Git 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)：超过 50 MiB 会警告，超过 100 MiB 会被阻止；大权重、数据及视频可使用 LFS 或合适的附件存储。

## 检查并上传

只添加实际存在的工程路径。下面假设已放入 Jetson 和 STM32 工程：

```bash
git status --short
git add software/jetson firmware/stm32
git diff --cached --stat
git diff --cached
git commit -m "Add existing Jetson and STM32 implementations"
git push -u origin codex/import-runtime-code
```

在 GitHub 为该分支创建拉取请求，核对后合并到 `main`。先看暂存差异，确认没有密码、令牌、本机私有路径或无关文件。如果远端已有更新，先拉取并解决冲突，不使用强制推送。

`.bin/.hex/.elf` 是构建产物，默认忽略；真实 `.c/.h/.ioc` 与工程配置不会因此被忽略。数据样例和模型索引也应单独添加并检查。
