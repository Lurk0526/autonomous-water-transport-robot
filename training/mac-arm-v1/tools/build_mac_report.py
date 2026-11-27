"""Build an auditable Chinese report from this completed local experiment."""
from pathlib import Path
import base64
import csv
from datetime import datetime
import hashlib
import html
import json
import math
import os
from statistics import mean
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(ROOT / 'outputs/.cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = ROOT / 'outputs/mac-run01-20261117-report'
OUT.mkdir(exist_ok=True)
RUN = ROOT / 'outputs/mac-run01-20261117'
def read(path):
    return json.loads((ROOT / path).read_text())

host = read('outputs/mac-run01-20261117-host.json')
execution = read('outputs/mac-run01-20261117.execution.json')
selection = read('outputs/mac-run01-20261117/pretest-selection.json')
run = read('outputs/mac-run01-20261117/run.json')
data = {
    'baseline': read('outputs/mac-holdout-baseline200-20261117/evaluation.json'),
    'best': read('outputs/mac-holdout-best200-20261117/evaluation.json'),
    'final': read('outputs/mac-holdout-final200-20261117/evaluation.json'),
}
reference = read('outputs/mac-baseline200-20261117/evaluation.json')
upstream = read('validation/baseline200/evaluation.json')
keys = ['baseline', 'best', 'final']
labels = {'baseline': '控制基线', 'best': '验证选出的 best_model', 'final': '训练结束的 final'}
enlabels = ['Baseline', 'Validation-selected', 'Final']
for name, d in data.items():
    assert d['episodes'] == len(d['trials']) == 200
    assert d['successes'] == sum(t['is_success'] for t in d['trials'])
    assert d['successes'] + sum(d['failures'].values()) == 200
    for key in ['config_sha256', 'model_sha256', 'source_sha256']:
        assert d['provenance'][key] == run['provenance'][key]
    assert np.array_equal([t['parking_xy_yaw'] for t in d['trials']],
                          [t['parking_xy_yaw'] for t in data['baseline']['trials']])
assert execution['exit_code'] == 0 and not run['smoke_only']

paired = {}
for name in ['best', 'final']:
    base = np.array([t['is_success'] for t in data['baseline']['trials']])
    policy = np.array([t['is_success'] for t in data[name]['trials']])
    gained, lost = int(np.sum(policy & ~base)), int(np.sum(base & ~policy))
    n = gained + lost
    pvalue = min(1., 2 * sum(math.comb(n, j) for j in range(min(gained, lost) + 1)) / 2**n) if n else 1.
    paired[name] = {'policy_only_successes': gained, 'baseline_only_successes': lost,
                    'difference_percentage_points': (data[name]['success_rate'] - data['baseline']['success_rate']) * 100,
                    'exact_two_sided_p': pvalue}

events = EventAccumulator(str(RUN / 'tensorboard/PPO_1'), size_guidance={'scalars': 0})
events.Reload()
def scalar(tag):
    values = events.Scalars(tag)
    return np.array([v.step for v in values]), np.array([v.value for v in values])
validation = np.load(RUN / 'eval/evaluations.npz')
vsteps = validation['timesteps']
vreturn = validation['results'].mean(axis=1)
vsuccess = validation['successes'].mean(axis=1)
completed_episodes = 0
for p in RUN.glob('*.monitor.csv'):
    with p.open() as f:
        next(f)
        completed_episodes += sum(1 for _ in csv.DictReader(f))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'svg.fonttype': 'none'})
colors = ['#64748b', '#087f8c', '#df7844']
fig, axs = plt.subplots(2, 2, figsize=(12, 8.3), layout='constrained')
x, y = scalar('rollout/ep_rew_mean')
axs[0, 0].plot(x, y, color=colors[1], label='Training: latest 100 episodes')
axs[0, 0].plot(vsteps, vreturn, 'o-', color=colors[2], label='Validation: 20 episodes / check')
axs[0, 0].set(title='Return did not improve consistently', xlabel='Environment steps', ylabel='Mean episodic return')
axs[0, 0].legend(fontsize=8)
x, y = scalar('rollout/success_rate')
axs[0, 1].plot(x, y * 100, color=colors[1], label='Training: latest 100 episodes')
axs[0, 1].plot(vsteps, vsuccess * 100, 'o-', color=colors[2], label='Validation: 20 episodes / check')
axs[0, 1].axhline(95, ls='--', color='#b42318', lw=1, label='Provisional target: 95%')
axs[0, 1].set(title='Training and validation are diagnostic', xlabel='Environment steps', ylabel='Success (%)', ylim=(0, 102))
axs[0, 1].legend(fontsize=8, loc='lower right')
rates = np.array([data[k]['success_rate'] for k in keys]) * 100
lo = rates - np.array([data[k]['success_rate_wilson95'][0] for k in keys]) * 100
hi = np.array([data[k]['success_rate_wilson95'][1] for k in keys]) * 100 - rates
axs[1, 0].bar(enlabels, rates, color=colors, width=.6)
axs[1, 0].errorbar(enlabels, rates, yerr=[lo, hi], fmt='none', color='#1e293b', capsize=5)
for i, k in enumerate(keys):
    axs[1, 0].text(i, rates[i] + hi[i] + 2, f'{data[k]["successes"]}/200', ha='center')
axs[1, 0].axhline(95, ls='--', color='#b42318', lw=1)
axs[1, 0].set(title='New holdout: no meaningful gain', ylabel='Success (%) with 95% Wilson interval', ylim=(0, 105))
reasons = sorted(set().union(*(data[k]['failures'] for k in keys)),
                 key=lambda r: sum(data[k]['failures'].get(r, 0) for k in keys), reverse=True)
short = {'unexpected_collision': 'Collision', 'lift_hold_not_verified': 'Lift / hold',
         'joint_tracking_limit': 'Tracking', 'final_placement_not_verified': 'Final placement',
         'grasp_contact_missing': 'No grip contact', 'support_missing_release_inhibited': 'Support',
         'dropped_bowl': 'Dropped bowl'}
pos = np.arange(len(reasons))
for i, k in enumerate(keys):
    axs[1, 1].barh(pos + (i-1)*.24, [data[k]['failures'].get(r, 0) for r in reasons],
                  height=.23, label=enlabels[i], color=colors[i])
axs[1, 1].set_yticks(pos, [short.get(r, r) for r in reasons])
axs[1, 1].invert_yaxis()
axs[1, 1].set(title='Failure counts on identical parking poses', xlabel='Failed episodes out of 200')
axs[1, 1].legend(fontsize=8)
for ax in axs.flat:
    ax.grid(axis='y' if ax is not axs[1, 1] else 'x', alpha=.15)
    ax.set_axisbelow(True)
fig.savefig(OUT / 'training-results.png', dpi=170, facecolor='white')
fig.savefig(OUT / 'training-results.svg', facecolor='white')
plt.close(fig)

def pct(x): return f'{x*100:.1f}%'
table_rows = []
for k in keys:
    d = data[k]
    lo, hi = d['success_rate_wilson95']
    table_rows.append([labels[k], f'{d["successes"]}/200', pct(d['success_rate']),
                       f'{pct(lo)}—{pct(hi)}', f'{d["reachable_episodes"]}/200',
                       f'{mean(t["return"] for t in d["trials"]):.2f}'])
fail_zh = {'unexpected_collision': '非预期碰撞', 'lift_hold_not_verified': '抬升保持未通过',
           'joint_tracking_limit': '关节跟踪超限', 'final_placement_not_verified': '最终放置未通过',
           'grasp_contact_missing': '夹持接触缺失', 'support_missing_release_inhibited': '承托不足、禁止松爪',
           'dropped_bowl': '掉碗'}
failure_rows = [[fail_zh.get(r, r), *[str(data[k]['failures'].get(r, 0)) for k in keys]] for r in reasons]
def mdtable(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])
baseline, best, final = (data[k] for k in keys)
best_pair = paired['best']
summary = {
    'generated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'conclusion': '本轮 10 万步 PPO 未显示出相对控制基线的可靠收益；未达到 95% 候选仿真目标。',
    'host': host, 'training_execution': execution, 'model_selection': selection,
    'completed_training_episodes': completed_episodes,
    'holdout': {k: {key: value for key, value in d.items() if key not in ['trials', 'provenance']}
                for k, d in data.items()},
    'paired_comparisons': paired,
    'same_config_model_source_hashes': True,
    'same_holdout_parking_poses': True,
    'hardware_ready': False,
}
(OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

text = f'''# Mac 机械臂抓碗首轮训练报告

2026 年 11 月 17 日 · AstraGO 候选仿真模型 · 本机实测

**本轮训练完成，但没有获得可靠的成功率收益。** 新留出测试中，控制基线为 {pct(baseline['success_rate'])}，验证选出的权重为 {pct(best['success_rate'])}，最终权重为 {pct(final['success_rate'])}；均未达到指南暂定的 95% 目标。训练程序跑通、GUI 能显示与策略已经可靠抓碗是不同层面的结果。

## 1. 本次实际完成的工作

- 安装验收的 11 项测试全部通过；本轮先完成 1024 步、2 环境短训，再从新初始化策略开始正式训练，没有续用短训权重。
- 正式训练请求 100000 步，实际保存 {selection['models']['final.zip']['num_timesteps']} 步。PPO 每轮收集 2×256=512 步，按完整采样批次结束，因此多出 352 步。
- 正式训练开始于 {execution['started_at'][11:19]}，结束于 {execution['ended_at'][11:19]}（北京时间），耗时 {execution['wall_seconds']:.1f} 秒，约 {execution['wall_seconds']/60:.2f} 分钟。该时间包含训练内验证与保存；基线及 GUI 预览曾并行运行，因此不作为纯训练吞吐基准。
- 完整结束训练回合 {completed_episodes} 个。保留 final、best_model、每 10000 步检查点、CSV、TensorBoard 和命令执行记录。
- 额外完成指南默认种子的 200 次本机基线复现，以及新留出种子上的基线、best、final 各 200 次测试，共 800 次评估回合。GUI 预览回合未纳入这些统计。

## 2. 运行环境与实验条件

| 项目 | 实测或本次设置 |
| --- | --- |
| 电脑 | Apple M2 Pro，16 GiB 内存，macOS 27.3，原生 arm64 |
| Python / MuJoCo | {host['python']} / {host['packages']['mujoco']} |
| PyTorch / SB3 / Gymnasium | {host['packages']['torch']} / {host['packages']['stable-baselines3']} / {host['packages']['gymnasium']} |
| 计算 | CPU，2 个训练环境；训练种子 42 |
| 网络与 PPO | 64×64；学习率 3e-4；gamma=0.99；每环境 256 步；每次 5 轮更新 |
| 停车随机范围 | x、y 各 ±25 mm，yaw ±15°；这些范围未由现场停车测量确定 |
| 载荷与感知 | 碗水等效刚体总质量 0.65 kg；坐标噪声和延迟估计器，未接真实 RGB |
| 策略输出 | 三维末端位置残差，最大 ±10 mm；夹爪开合仍由阶段控制 |

模型、源代码和配置在本轮训练与三组留出评估中的 SHA256 一致，没有为提高成功率放宽碰撞、限位、摩擦或成功判定。

## 3. 独立留出评估

采用种子 20000—20199。三组测试逐回合停车姿态完全一致，200 回合均保留在分母中。该种子组未用于本轮训练、自动验证或参数调整。主报告使用按自动验证平均回报选出的 best_model；final 作为并列对照，未根据留出结果重新调参或重选检查点。

{mdtable(['方法', '成功次数', '成功率', '95% Wilson 区间', '可达次数', '平均回报'], table_rows)}

best_model 来自 {selection['validation_best_step']} 步，训练内该次 20 回合验证成功率为 {pct(selection['validation_best_success_rate'])}、平均回报为 {selection['validation_best_mean_return']:.2f}。留出测试仅比基线多成功 {best['successes']-baseline['successes']} 回合，差 {paired['best']['difference_percentage_points']:+.1f} 个百分点，不能据此宣称学习策略优于基线。配对比较中，策略独有成功 {best_pair['policy_only_successes']} 次、基线独有成功 {best_pair['baseline_only_successes']} 次；探索性精确双侧配对二项检验 p={best_pair['exact_two_sided_p']:.3f}，不支持稳定收益。

最终权重比基线少成功 {baseline['successes']-final['successes']} 回合，低 {abs(paired['final']['difference_percentage_points']):.1f} 个百分点。当前证据说明追加步数没有自动带来更好结果；单次训练种子和每次 20 回合验证不足以判定具体原因，也不足以认定已经收敛或过拟合。

## 4. 曲线与失败分布

![训练曲线、留出对照及失败分布](training-results.png)

曲线中的训练成功率是最近 100 个已结束训练回合的统计；验证每次 20 回合。它们用于诊断，结论以上表留出评估为准。每次自动验证延续其随机数序列，并非固定同一组 20 个场景，因此验证折线波动同时包含策略变化和场景抽样差异。

{mdtable(['失败原因', '基线', 'best_model', 'final'], failure_rows)}

best_model 的碰撞次数有所减少，但抬升保持、跟踪和放置仍频繁失败；final 的最终放置失败明显增多。所有测试目标在候选模型下可达，说明本轮主要问题不能用“目标可达”来排除。统计按程序最终记录的终止原因计数，多个条件同时触发时并非完整的根因归类。

## 5. 复现差异与界面处理

指南默认种子 10000—10199 在本机基线为 {reference['successes']}/200（{pct(reference['success_rate'])}），随包 Linux 基线为 {upstream['successes']}/200（{pct(upstream['success_rate'])}）。本轮没有更改物理配置；跨平台结果未逐回合一致，具体原因尚未定位，因此正式比较均使用同一台 Mac 的新留出结果。

训练中按用户要求打开了 MuJoCo 策略预览，自动在回合边界加载最新检查点，训练结束后切换为 final.zip。它是单独环境的实时策略演示，不是训练工作进程的原始采样画面，也不计入留出测试。

离线包原始 TensorBoard 入口因 setuptools 不再提供 pkg_resources 而启动失败。已新增 tools/tensorboard_compat.py，针对本包 TensorBoard 实际使用的入口点发现和版本比较接口做离线适配，未修改已安装库、锁文件或训练逻辑。曲线服务已确认显示，地址为 http://127.0.0.1:6006/。这暴露了离线包的 GUI 辅助依赖兼容问题，pip check 通过并不能覆盖它。

## 6. 建议下一步与结果边界

先按指南核对关节刚体分组、零位与限位、MG23 真实联动及接触高度，再用独立实验目录进行接触搜索和基线对照。当前 61% 左右的控制基线仍不可靠，直接延长 PPO 训练缺少收益证据。几何和执行器条件合理后，再以多个训练种子重复验证；若依据本报告调整参数，种子 20000—20199 应转为开发集，下一轮需另设新的最终留出种子。

本模型仍采用 MG23 四指径向联动近似、未标定关节与泡沫参数、带噪坐标估计器和刚体水载荷，没有真实相机闭环、液面晃动或失水量仿真。接触证据按 20 Hz 控制采样检查，未连续认证所有物理子步峰值。本次结果不构成真实机械臂、满水防洒水或整车自主运输验收；hardware_ready 始终为 false。

## 7. 文件与复现入口

- 正式训练：outputs/mac-run01-20261117/；权重为 best_model.zip 和 final.zip。
- 命令、起止时间和完整日志：outputs/mac-run01-20261117.execution.json / .execution.log。
- 新留出原始结果：outputs/mac-holdout-baseline200-20261117/evaluation.json、outputs/mac-holdout-best200-20261117/evaluation.json、outputs/mac-holdout-final200-20261117/evaluation.json。
- 本报告数据汇总：summary.json；图片为 training-results.png / training-results.svg。
- 配置 SHA256：{run['provenance']['config_sha256']}。
- 模型 SHA256：{run['provenance']['model_sha256']}。
- 源码 SHA256：{run['provenance']['source_sha256']}。

在包根目录重新打开预览和曲线：

```bash
.venv/bin/mjpython tools/live_training_viewer.py --run outputs/mac-run01-20261117
.venv/bin/python tools/tensorboard_compat.py --logdir outputs/mac-run01-20261117/tensorboard --host 127.0.0.1 --port 6006 --load_fast false
```
'''
(OUT / 'Mac首轮训练报告.md').write_text(text)

# A self-contained HTML copy renders the same content without network assets.
import markdown
image_data = base64.b64encode((OUT / 'training-results.png').read_bytes()).decode()
body = markdown.markdown(text, extensions=['tables', 'fenced_code'])
body = body.replace('src="training-results.png"', 'src="data:image/png;base64,' + image_data + '"')
page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Mac 首轮训练报告</title>
<style>body{margin:0;background:#f2f5f7;color:#172b3a;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}
main{max-width:1060px;margin:30px auto;padding:38px 48px;background:white;border-radius:12px;box-shadow:0 4px 28px #1231}
h1{font-size:30px;line-height:1.35}h2{font-size:21px;margin-top:38px;color:#087f8c}strong{color:#9b3526}table{width:100%;border-collapse:collapse;font-size:14px;margin:20px 0}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #dce3e8}th{background:#eef5f6}img{max-width:100%;height:auto}pre{overflow:auto;background:#eef2f5;padding:16px;border-radius:6px}li{margin:6px 0}code{overflow-wrap:anywhere}p{overflow-wrap:anywhere}@media(max-width:700px){main{padding:20px;margin:0;border-radius:0}table{display:block;overflow:auto}h1{font-size:25px}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0}h2{break-after:avoid}table,img{break-inside:avoid}}</style>
<main>''' + body + '</main></html>'
(OUT / 'Mac首轮训练报告.html').write_text(page)
print(json.dumps({'report_dir': str(OUT), 'success_rates': {k: data[k]['success_rate'] for k in keys},
                  'training_episodes': completed_episodes, 'paired': paired}, ensure_ascii=False, indent=2))
