"""One-command CPU preflight. Passing it never certifies a physical grasp."""
import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import unittest
from demo import run as run_demo
from screening import sweep
from actuator_screening import audit as actuator_audit
from reward_audit import run_attacks
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_bowl_transport*.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (out/'test-results.txt').write_text(stream.getvalue())
    config=json.loads((HERE/'scenario.json').read_text())
    timeline=run_demo(config,out)
    screen=sweep();actuator=actuator_audit();rewards=run_attacks()
    for name,data in [('screening',screen),('actuator-screening',actuator),('reward_audit',rewards)]:
        (out/(name+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
    local_ok=(result.wasSuccessful() and timeline['final_phase']=='DONE' and screen['all_checks_passed']
              and actuator['all_checks_passed'] and rewards['all_passed'])
    # No inferred 'passed' entries for physical assets or measurements.
    blockers=[
        ('robot_asset','关节轴、限位、驱动、夹爪耦合及真实碰撞资产未验证'),
        ('grasp_and_payload','156 mm 包络、湿碗夹持、650 g 持续承载与回差未实测'),
        ('observability','相机/手眼、时间对齐、底盘制动、夹持/承托证据链未验收'),
        ('fluid','空碗质量、内腔、溢流模型与称重误差未测；实际洒水量未知'),
        ('scoring','赛事洒水容忍度及时间/洒水评分规则未确认'),
        ('baseline','未证明确定性控制不足，也没有真实抓放基线和留出工况结果'),
    ]
    report={'schema_version':2,'kind':'local_logic_numerical_reward_preflight',
            'local_checks_passed':local_ok,'test_methods_run':result.testsRun,
            'test_failures':len(result.failures),'test_errors':len(result.errors),
            'numerical_checks':len(screen['checks']),'actuator_checks':len(actuator['checks']),
            'reward_attack_checks':len(rewards['checks']),
            'training_rental_ready':False,'physical_grasp_validated':False,
            'fluid_simulated':False,'hardware_command_sent':False,
            'blockers':[{'id':n,'status':'not_verified','detail':d} for n,d in blockers],
            'screening_counterexamples':{'simultaneous_worse':screen['overlap_worse_count'],'total':screen['scenario_count'],
                 'meaning':'Hypothetical single-mode sensitivity only; NOT observed spill or evidence to select a real trajectory.'},
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(HERE.glob('*.py'))+sorted((ROOT/'tests').glob('test_bowl_transport*.py'))+[HERE/'scenario.json']}}
    (out/'preflight.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    validation_path=HERE/'validation.json'
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    validation.pop('gate_tests_passed',None)
    validation.update(schema_version=2,local_test_methods_passed=result.testsRun if result.wasSuccessful() else None,
                      local_checks_passed=local_ok,training_rental_ready=False,
                      current_validation_report='output/preflight.json',
                      usd_check_scope='Prior USD structure check only; no dynamic robot or fluid validation')
    validation_path.write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    rows=''.join(f'<tr><td>{html.escape(d)}</td><td class="wait">待验证</td></tr>' for _,d in blockers)
    checks=''.join(f'<li>{html.escape(k)}：{"通过" if v else "失败"}</li>' for k,v in rewards['checks'].items())
    sensitivity=''.join(f'<tr><td>{r["frequency_hz"]}</td><td>{r["damping"]}</td><td>{r["arm_duration_s"]}</td><td>{r["simultaneous_mm"]:.3f}</td><td>{r["after_braking_mm"]:.3f}</td></tr>' for r in screen['scenarios'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>满水碗搬运 · 本机审计 v0.2</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#243b4a;background:#f4f7f9;margin:0}main{max-width:1000px;margin:36px auto;padding:0 22px}h1{font-size:30px;line-height:1.3}h2{font-size:21px;margin-top:30px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card,section{background:white;border:1px solid #dbe3e7;border-radius:12px;padding:20px;margin:15px 0}.good{color:#087a58}.wait{color:#ab520d}.big{font-size:26px;font-weight:700}.note{border-left:4px solid #e7a63c;background:#fff8e9;padding:15px}a{color:#116da1}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #dfe7ec}code{background:#eef2f4;padding:2px 5px}summary{cursor:pointer}li{margin:8px 0}@media(max-width:600px){.cards{grid-template-columns:1fr}}</style><main><h1>满水碗搬运 · 本机审计 v0.2</h1><p>已知：碗口 156 mm · 高约 60 mm · 碗与水总重 650 g · 水至碗口。空碗/水质量仍未知。</p>'''
    page+=f'<div class="cards"><div class="card"><div>本机逻辑 / 数值 / 奖励检查</div><div class="big {"good" if local_ok else "wait"}">{"通过" if local_ok else "未通过"}</div><p>{result.testsRun} 项测试；{len(rewards["checks"])} 项奖励攻击检查。</p></div><div class="card"><div>训练服务器租赁准入</div><div class="big wait">尚未具备条件</div><p>建议暂不租长时训练服务器。</p></div></div>'
    page+='<p class="note">本机未运行完整机器人动力学、物理流体或强化学习。合成成功信号只能检查程序；数学模型不能输出真实洒水量。本次没有发送任何实机运动指令。</p>'
    page+='<section><h2>此次纠正的核心问题</h2><ol><li>下降定位失效、抬起后夹持失效、行驶路径受阻，现在会锁存 HOLD 请求；HOLD 的实机受控停车/持载恢复仍需实现。</li><li>夹持、承托与松脱必须连续确认；时戳重放、大采样缺口、陈旧数据不能累计稳定时间。</li><li>行驶中展开默认关闭。先建立停车基线，再对限定路段做等条件对照；并非否定最终提前准备的目标。</li><li>低位携带、缓慢动作和绝对水平都不是直接优化目标；整段运动和真实液体损失才决定优劣。</li></ol></section>'
    page+=f'<section><h2>“制动与回收同时做”是否一定更好？</h2><p>在 {screen["scenario_count"]} 组<strong>假设参数</strong>中，{screen["overlap_worse_count"]} 组同时执行的简化边缘响应更大。没有标定频率与阻尼，不能用这些数字选实机轨迹或估算洒水克数。</p><p>修复了旧演示把静态液面偏移与含静态偏移的响应重复相加的问题；目前使用有符号的总响应，并检查时间步收敛。零余量下溢流会改变水量，本小幅模型未覆盖这一过程。</p><details><summary>查看敏感性测试参数与响应（mm，非洒水量）</summary><table><tr><th>频率/Hz</th><th>阻尼比</th><th>臂动作时间/s</th><th>同时</th><th>制动后</th></tr>{sensitivity}</table></details></section>'
    page+='<section><h2>奖励函数：先审计，后决定是否训练</h2><p>候选数值只用于记账验证，未经机器人标定。成功要有独立证据和明确洒水阈值；实际水量/洒水未知时完整奖励为 null。失败、不可逆水损采用固定回合代价，推迟发生不会得到折扣优惠；阶段势函数正确清零，重复接近不赚分。</p><p>实际评价先看约束是否合格，再比相同条件的洒水量和时间。不同初水量、路线或证据来源不混比。</p><details><summary>奖励漏洞检查</summary><ul>'+checks+'</ul></details></section>'
    page+='<section><h2>租训练服务器前仍缺什么</h2><table>'+rows+'</table><p>同一工况下独立重复 20 次零失败，按二项模型也只能给出约 13.9% 的单侧 95% 失败率上界，不能据此声称 99% 可靠。不同工况需分别统计。</p></section>'
    page+='<section><h2>重跑与证据</h2><p>项目根目录运行 <code>python3 sim/bowl_transport/preflight.py</code>。加 <code>--require-training-ready</code> 时，只要实际准入条件未满足就返回非零退出码。</p><p><a href="timeline.html">合成动作顺序</a> · <a href="preflight.json">验收结果与代码哈希</a> · <a href="test-results.txt">测试日志</a> · <a href="reward_audit.json">奖励审计</a> · <a href="screening.json">敏感性数据</a> · <a href="actuator-screening.json">执行器延迟示例</a></p></section></main></html>'
    (out/'report.html').write_text(page,encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','blockers')},ensure_ascii=False,indent=2))
    print('Report:',out/'report.html')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'output')
    parser.add_argument('--require-training-ready',action='store_true')
    args=parser.parse_args();r=run(args.out)
    sys.exit(1 if not r['local_checks_passed'] else 3 if args.require_training_ready and not r['training_rental_ready'] else 0)
