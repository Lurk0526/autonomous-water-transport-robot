"""Receiving-Mac workflow: verify first, then bounded training when evidence allows.

This driver never sends robot commands. Packaging host must not run it. A failed
vision stage does not get repaired with truth, and arm contact faults block scale.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def _captured_provenance(value):
    return value.get('provenance',value)


def _verify_current_package(expected):
    """Recheck source/model and configuration before every training stage."""
    from .training_gate import current_package_provenance
    now=current_package_provenance()
    for key in ('package_generation','source_sha256','model_sha256'):
        if now.get(key)!=expected.get(key):raise RuntimeError('Package changed after verification: '+key)
    # Match cli.provenance's canonical config hash, with the real current YAML.
    import yaml
    config=yaml.safe_load((ROOT/'config/train.yaml').read_text())
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()
    if digest!=expected.get('config_sha256'):raise RuntimeError('Training configuration changed after verification')


def _training_evaluation_guard(run_folder,mobile,log,expected):
    """Audit every saved development evaluation, not just a selected winner.

    Earlier exploration policies may have low success.  This check blocks scale
    for physical anomalies and missing evidence; only the externally repeated
    best/final development evaluations establish the success-rate gate.
    """
    from .training_gate import build_training_gate
    folder=Path(run_folder)
    run=read(folder/'run.json')
    integrity=read(folder/'source-integrity.json')
    reports=[]
    paths=sorted((folder/'eval').glob('*.json'))
    for path in paths:
        gate=build_training_gate(read(path),mobile,log,expected_provenance=expected)
        check=gate['checks']['baseline']
        reports.append({'path':str(path),'serious_event_count':check['serious_event_count'],
            'guard_evidence_complete':check['guard_evidence_complete'],
            'provenance_valid':gate['checks']['baseline_provenance']['valid'],
            'evidence_valid':not any(reason['scope']=='arm' and reason['code']!='baseline_success_below_gate'
                for reason in gate['reasons']),
            'guard_config_valid':not any(reason['scope']=='arm' and reason['code'] in
                ('guard_threshold_missing_or_relaxed','foam_contact_guard_disabled') for reason in gate['reasons'])})
    valid=bool(paths and run.get('status')=='completed' and integrity.get('unchanged_during_run') is True
        and all(item['serious_event_count']==0 and item['guard_evidence_complete']
            and item['provenance_valid'] and item['guard_config_valid'] and item['evidence_valid'] for item in reports))
    return {'valid':valid,'run_status':run.get('status'),'source_integrity':integrity,
        'evaluations':reports,'scope':'Saved development evaluations only; training-rollout fault telemetry is separate and must be retained.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--large-if-ready',action='store_true')
    p.add_argument('--skip-benchmark',action='store_true');p.add_argument('--vision-device',choices=['cpu','mps'],default='cpu')
    p.add_argument('--output',type=Path);a=p.parse_args()
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise SystemExit('Run this workflow only on the receiving native arm64 Mac.')
    out=a.output or ROOT/'outputs'/('mac-v3-check-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out=out.resolve()
    if out.exists():raise SystemExit('Use a fresh output folder; do not overwrite earlier evidence')
    logs=out/'logs';logs.mkdir(parents=True);stages=[];training_started=False
    py=sys.executable
    def command(label,argv,allow_failure=False):
        logfile=logs/(label+'.log');print(f'{label}: {logfile}',flush=True)
        try:
            with logfile.open('w') as f:result=subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            code=result.returncode
        except BaseException as e:
            stages.append({'label':label,'command':argv,'status':'interrupted_or_error','error':repr(e)});write(out/'workflow-status.json',stages);raise
        stages.append({'label':label,'command':argv,'exit_code':code,'log':str(logfile)})
        write(out/'workflow-status.json',stages)
        if code and not allow_failure:raise RuntimeError(f'{label} failed ({code}); preserve logs and package the return. No subsequent larger training stage will start.')
        return code
    def arm(label,*args,allow_failure=False):return command(label,[py,'-m','arm_training.cli',*map(str,args)],allow_failure)
    try:
        command('tests',[py,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v'])
        arm('inspect','inspect','--output',out/'inspection.json')
        expected=_captured_provenance(read(out/'inspection.json'))
        _verify_current_package(expected)
        # Throughput is independent of grasp/contact acceptance.  Even a
        # rejected physical model still yields useful CPU/MPS timing evidence.
        device='cpu';envs=2;benchmark_ready=True
        if not a.skip_benchmark:
            import torch
            mps_available=bool(torch.backends.mps.is_available())
            benchmark_args=['benchmark','--experimental-model','--output',out/'benchmark']
            if mps_available:benchmark_args.append('--compare-mps')
            write(out/'benchmark-capabilities.json',{'mps_available':mps_available,
                'comparison_requested':mps_available,
                'reason':'Explicit CPU+MPS throughput comparison' if mps_available else 'MPS unavailable; explicitly benchmark CPU only, without silent device fallback.'})
            code=arm('benchmark',*benchmark_args,allow_failure=True)
            report=out/'benchmark/benchmark.json'
            b=read(report) if report.exists() else {}
            benchmark_ready=bool(not code and b.get('completed') is True)
            if benchmark_ready:
                envs=b.get('fastest_measured_cpu_envs')
                ratio=b.get('mps_to_matching_cpu_throughput_ratio',0.)
                if type(envs) is not int or envs not in range(1,9) or not isinstance(ratio,(int,float)) or not math.isfinite(ratio):
                    benchmark_ready=False
                elif mps_available and ratio>=1.10:device='mps'
        selection={'device':device if benchmark_ready else None,'envs':envs if benchmark_ready else None,
            'policy_network':'64x64 MLP','physics_device':'cpu','benchmark_ready':benchmark_ready,
            'benchmark_skipped_explicitly':a.skip_benchmark,
            'reason':'Explicit skip uses CPU with 2 environments.' if a.skip_benchmark else
                'Use MPS only if measured sampling+updates throughput is at least 10% better; a failed comparison blocks training rather than silently falling back.'}
        write(out/'execution-selection.json',selection)
        for seed in [30003,40004,40010,40022]:arm(f'diagnose-{seed}','diagnose','--seed',seed,'--output',out/f'diagnose-{seed}')
        arm('baseline-dev100','baseline','--episodes',100,'--seed',30000,'--output',out/'baseline-dev100')
        # Compare the receiving Mac with its previous return; this is a replication set, not an untouched holdout.
        arm('baseline-replication200','baseline','--episodes',200,'--seed',40000,'--output',out/'baseline-replication200')
        mobile_result=None
        mobile_first_code=command('mobile-first3',[py,'-m','arm_training.mobile_cli','evaluate','--episodes','3','--seed','70000',
            '--vision-device',a.vision_device,'--output',str(out/'mobile-first3')],allow_failure=True)
        initial=out/'mobile-first3/mobile-evaluation.json'
        if initial.exists():
            mobile_result=initial
            r=read(initial)
            if mobile_first_code==0 and r.get('episodes')==3 and r.get('learned_vision') is True and r.get('vision_backend')=='ultralytics_yolo_segmenter' and all(r.get('stage_success_rates',{}).get(k,0)==1 for k in ['main_recognized','parking_selected','parked','wrist_localized']):
                mobile20_code=command('mobile-dev20',[py,'-m','arm_training.mobile_cli','evaluate','--episodes','20','--seed','70100',
                    '--vision-device',a.vision_device,'--output',str(out/'mobile-dev20')],allow_failure=True)
                if mobile20_code==0 and (out/'mobile-dev20/mobile-evaluation.json').exists():mobile_result=out/'mobile-dev20/mobile-evaluation.json'
        if mobile_result is None:
            # --mobile is deliberately required by training_gate.  Explicit
            # missing evidence permits the independent arm check to continue.
            mobile_result=out/'mobile-not-completed.json'
            write(mobile_result,{'episodes':0,'trials':[],'learned_vision':False,
                'vision_backend':None,'reason':'No completed learned-vision mobile result; inspect logs/mobile-first3.log'})
        gateargs=[py,'-m','arm_training.training_gate','--baseline',str(out/'baseline-dev100/evaluation.json'),
            '--tests-log',str(logs/'tests.log'),'--expected-provenance',str(out/'inspection.json'),'--output',str(out/'training-gate.json')]
        gateargs+=['--mobile',str(mobile_result)]
        command('training-gate',gateargs,allow_failure=True)
        gate=read(out/'training-gate.json')
        from .training_gate import build_training_gate
        mobile=read(mobile_result);test_log=(logs/'tests.log').read_text()
        replication_gate=build_training_gate(read(out/'baseline-replication200/evaluation.json'),mobile,test_log,expected_provenance=expected)
        write(out/'replication-training-gate.json',replication_gate)
        gate_ready=bool(gate['arm_residual_training_allowed'] and replication_gate['arm_residual_training_allowed'])
        if not gate_ready or not benchmark_ready:
            write(out/'training-decision.json',{'larger_training_started':False,
                'reason':'contact_or_baseline_gate_not_ready' if not gate_ready else 'requested_CPU_MPS_benchmark_failed_or_incomplete',
                'arm_gate_ready':gate_ready,'benchmark_ready':benchmark_ready,
                'next':'Return contact diagnostic windows and learned camera failure evidence; keep mobile development separate.'})
            print('Arm scale-up was not started: read training gates and execution-selection.json. Vision and parking evidence are preserved.',flush=True)
            return 2
        if not a.large_if_ready:
            write(out/'training-decision.json',{'larger_training_started':False,'reason':'checks_only_mode','gate_ready':True,'execution':selection})
            return 0
        _verify_current_package(expected)
        training_started=True
        arm('train-stage1-20k','train','--experimental-model','--steps',20000,'--seed',42,'--device',device,'--envs',envs,'--output',out/'train-stage1-20k')
        arm('stage1-dev100','evaluate','--experimental-model','--policy',out/'train-stage1-20k/best_model.zip','--device',device,
            '--episodes',100,'--seed',30000,'--output',out/'stage1-dev100')
        arm('stage1-final-dev100','evaluate','--experimental-model','--policy',out/'train-stage1-20k/final.zip','--device',device,
            '--episodes',100,'--seed',30000,'--output',out/'stage1-final-dev100')
        arm('stage1-paired','compare',out/'baseline-dev100/evaluation.json',out/'stage1-dev100/evaluation.json','--output',out/'stage1-paired.json')
        policy=read(out/'stage1-dev100/evaluation.json');baseline=read(out/'baseline-dev100/evaluation.json')
        _verify_current_package(expected)
        check=build_training_gate(policy,mobile,test_log,expected_provenance=expected)
        final_check=build_training_gate(read(out/'stage1-final-dev100/evaluation.json'),mobile,test_log,expected_provenance=expected)
        all_evaluations=_training_evaluation_guard(out/'train-stage1-20k',mobile,test_log,expected)
        nonregressing=policy['success_rate']>=baseline['success_rate']-.02
        advance_allowed=bool(check['arm_residual_training_allowed'] and final_check['arm_residual_training_allowed']
            and all_evaluations['valid'] and nonregressing)
        write(out/'stage1-decision.json',{'advance_to_100k':advance_allowed,'no_more_than_2pp_development_regression':nonregressing,
            'guard_check':check,'final_policy_guard_check':final_check,'all_saved_evaluation_guards':all_evaluations,
            'inference':'A scheduling decision only, not statistical evidence of learned improvement.'})
        if advance_allowed:
            _verify_current_package(expected)
            # Continue the separately verified final20k checkpoint for another
            # 80k. The earlier best checkpoint remains preserved for comparison.
            resumed_steps=read(out/'train-stage1-20k/run.json')['actual_total_timesteps']
            if type(resumed_steps) is not int or not 20000<=resumed_steps<30000:
                raise RuntimeError('Invalid verified stage1 final checkpoint step count')
            additional_steps=80000
            write(out/'stage2-budget.json',{'nominal_cumulative_target_steps':100000,'resumed_final_checkpoint_steps':resumed_steps,
                'requested_additional_steps':additional_steps,'rollout_rounding_expected':True,
                'actual_cumulative_steps_can_exceed_100000_by_rollout_rounding':True})
            arm('train-stage2-to100k','train','--experimental-model','--resume',out/'train-stage1-20k/final.zip',
                '--steps',additional_steps,'--seed',42,'--device',device,'--envs',envs,'--output',out/'train-stage2-to100k')
            arm('stage2-dev100','evaluate','--experimental-model','--policy',out/'train-stage2-to100k/best_model.zip','--device',device,
                '--episodes',100,'--seed',30000,'--output',out/'stage2-dev100')
            arm('stage2-final-dev100','evaluate','--experimental-model','--policy',out/'train-stage2-to100k/final.zip','--device',device,
                '--episodes',100,'--seed',30000,'--output',out/'stage2-final-dev100')
            arm('stage2-paired','compare',out/'baseline-dev100/evaluation.json',out/'stage2-dev100/evaluation.json','--output',out/'stage2-paired.json')
            _verify_current_package(expected)
            stage2_best=build_training_gate(read(out/'stage2-dev100/evaluation.json'),mobile,test_log,expected_provenance=expected)
            stage2_final=build_training_gate(read(out/'stage2-final-dev100/evaluation.json'),mobile,test_log,expected_provenance=expected)
            stage2_all=_training_evaluation_guard(out/'train-stage2-to100k',mobile,test_log,expected)
            stage2_accepted=bool(stage2_best['arm_residual_training_allowed'] and stage2_final['arm_residual_training_allowed'] and stage2_all['valid'])
            write(out/'stage2-decision.json',{'development_acceptance_passed':stage2_accepted,
                'best_policy_guard_check':stage2_best,'final_policy_guard_check':stage2_final,'all_saved_evaluation_guards':stage2_all,
                'fresh_holdout_has_not_run':True,'deployable':False})
        else:stage2_accepted=False
        write(out/'training-decision.json',{'larger_training_started':True,'stage2_started':advance_allowed,'execution':selection,
            'final_holdout_50000_to_50199':'untouched','end_to_end_mobile_policy_training':False})
        return 0 if stage2_accepted else 2
    except BaseException as error:
        write(out/'training-decision.json',{'larger_training_started':training_started,'workflow_completed':False,
            'reason':'workflow_interrupted_or_failed','error':f'{type(error).__name__}: {error}',
            'next':'Preserve all completed and partial evidence; do not resume a larger stage automatically.'})
        raise
    finally:
        print('Preserve all evidence at',out,flush=True)


if __name__=='__main__':raise SystemExit(main())
