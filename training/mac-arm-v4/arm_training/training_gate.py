"""Read-only evidence checks before bounded simulation training; no hardware I/O.

This module reads JSON and a unittest text log.  It never executes content from
these files, imports the simulator, runs tests, or starts training.  Passing this
gate is neither a hardware safety certificate nor proof of population accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from .fingerprints import model_fingerprint
import json
import math
from pathlib import Path
import re


GENERATION = 'mac-arm-v4'
HASH_KEYS = ('config_sha256', 'source_sha256', 'model_sha256')
REAL_SEGMENTERS = {'ultralytics_yolo_segmenter', 'ultralytics_yolo11n_seg',
                   'ultralytics_yolo', 'yolo11n-seg', 'target_bowl_yolo11n_seg'}
SERIOUS_TOKENS = ('hard_contact', 'closure_joint_limit', 'closure_coupling_limit',
                  'nonfinite', 'non_finite', 'physics_warning', 'unexpected_collision',
                  'table_collision', 'platform_collision', 'joint_tracking_limit',
                  'tracking_limit', 'joint_limit', 'dropped_bowl', 'payload_loss')
SERIOUS_TOKENS += ('foam_contact_force_limit','rack_force_limit','closed_chain_loop_error')
ROOT = Path(__file__).resolve().parents[1]


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _reason(scope, code, detail):
    return {'scope': scope, 'code': code, 'detail': detail}


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _contains_nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def current_package_provenance():
    """Fingerprint source/model bytes with the same algorithm as cli.provenance."""
    digest = hashlib.sha256()
    files = sorted((ROOT / 'arm_training').glob('*.py'))
    if not files:
        raise ValueError('current_package_sources_missing')
    for source in files:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return {'package_generation': GENERATION,
            'source_sha256': digest.hexdigest(),
            'model_sha256': model_fingerprint(ROOT)}


def _provenance_check(record, scope, expected):
    reasons, value = [], record.get('provenance')
    if not isinstance(value, dict):
        return {'valid': False, 'record': None}, [_reason(scope, 'provenance_missing', 'Evaluation needs captured package/config/source/model provenance.')]
    if value.get('package_generation') != GENERATION:
        reasons.append(_reason(scope, 'wrong_package_generation', 'Only mac-arm-v4 evaluation evidence is accepted.'))
    for key in HASH_KEYS:
        if not _hash(value.get(key)):
            reasons.append(_reason(scope, 'invalid_' + key, 'A complete lowercase SHA-256 is required.'))
        elif expected.get(key) is not None and value[key] != expected[key]:
            reasons.append(_reason(scope, 'mismatched_' + key, 'Evaluation differs from the expected package/config fingerprint.'))
    config = value.get('config')
    if not isinstance(config, dict) or _contains_nonfinite(config):
        reasons.append(_reason(scope, 'captured_config_missing_or_invalid', 'Captured configuration is required, including evidence thresholds.'))
    else:
        calculated = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if calculated != value.get('config_sha256'):
            reasons.append(_reason(scope, 'captured_config_hash_mismatch', 'Embedded configuration does not match its fingerprint.'))
    return {'valid': not reasons, 'record': {key: value.get(key) for key in ('package_generation', *HASH_KEYS)}}, reasons


def _tests_check(log):
    matches = re.findall(r'^Ran\s+(\d+)\s+tests?\s+in\s+[^\r\n]+', log, re.MULTILINE)
    count = int(matches[-1]) if matches else 0
    finished = re.search(r'^OK\s*$', log, re.MULTILINE) is not None
    failure = re.search(r'\b(?:FAILED|ERROR)\b|^Traceback\b', log, re.MULTILINE) is not None
    skipped = re.search(r'^OK\s*\([^\r\n]*skipped\s*=\s*[1-9]', log, re.MULTILINE) is not None
    # Empty logs or a hand-copied standalone OK are not completed unittest runs.
    valid = count > 0 and finished and not failure and not skipped
    return {'valid': valid, 'test_count': count, 'completed_ok': finished,
            'failure_marker_seen': failure, 'skipped_tests_seen': skipped,
            'log_sha256': hashlib.sha256(log.encode()).hexdigest()}


def _threshold_check(config):
    """Do not admit a run by weakening the original physical fault thresholds."""
    reasons = []
    limits = [('task', 'joint_tracking_limit_rad', .25),
              ('task', 'gate_tracking_rad', .06),
              ('gripper', 'closure_limit_tolerance_m', .001),
              ('gripper', 'closure_sync_tolerance_m', .003)]
    for section, key, cap in limits:
        section_value = config.get(section)
        value = section_value.get(key) if isinstance(section_value, dict) else None
        if not _finite(value) or not 0 < value <= cap + 1e-12:
            reasons.append(_reason('arm', 'guard_threshold_missing_or_relaxed',
                f'{section}.{key} must be measured/retained within the provisional cap {cap}; increasing it cannot bypass a failed gate.'))
    task = config.get('task') if isinstance(config.get('task'), dict) else {}
    if config.get('compliant_grasp',{}).get('allow_bounded_mixed_contact',False):
        reasons.append(_reason('arm','unvalidated_mixed_contact_mode','The experimental mixed contact diagnostic is not a verified foam grasp.'))
    if task.get('contact_supervisor_mode') != 'foam_required':
        reasons.append(_reason('arm', 'foam_contact_guard_disabled', 'The v4 baseline must retain foam_required contact supervision.'))
    return reasons


def _serious_events(record, path='evaluation'):
    """Collect diagnostic evidence, without interpreting any strings as commands."""
    issues = []
    if isinstance(record, dict):
        for key, value in record.items():
            item_path = path + '.' + str(key)
            if key in ('first_collision', 'first_physics_anomaly') and value is not None:
                issues.append({'path': item_path, 'kind': key, 'detail': value})
            if key == 'peak_gripper_contact' and isinstance(value, dict):
                force = value.get('force_n')
                if value.get('foam') is False and _finite(force) and force >= .5:
                    issues.append({'path': item_path, 'kind': 'hard_contact_peak', 'detail': value})
            if key in ('q_tracking_error_rad', 'max_q_tracking_error_rad') and _finite(value) and value > .25 + 1e-12:
                issues.append({'path': item_path, 'kind': 'joint_tracking_limit', 'detail': value})
            if key in ('physics_warning_count', 'nonfinite_count', 'collision_count') and _finite(value) and value > 0:
                issues.append({'path': item_path, 'kind': key, 'detail': value})
            if key in ('failures', 'guard_counts') and isinstance(value, dict):
                for name, count in value.items():
                    if _finite(count) and count > 0 and any(token in str(name).lower() for token in SERIOUS_TOKENS):
                        issues.append({'path': item_path + '.' + str(name), 'kind': str(name), 'detail': count})
            if isinstance(value, str) and key in ('failure_reason', 'reason', 'kind', 'fault', 'anomaly'):
                if any(token in value.lower() for token in SERIOUS_TOKENS):
                    issues.append({'path': item_path, 'kind': value, 'detail': value})
            if isinstance(value, (dict, list)):
                issues.extend(_serious_events(value, item_path))
            elif isinstance(value, float) and not math.isfinite(value):
                issues.append({'path': item_path, 'kind': 'nonfinite', 'detail': repr(value)})
    elif isinstance(record, list):
        for index, value in enumerate(record):
            issues.extend(_serious_events(value, f'{path}[{index}]'))
    return issues


def _baseline_check(record):
    reasons = []
    episodes, trials = record.get('episodes'), record.get('trials')
    if type(episodes) is not int or episodes < 100:
        reasons.append(_reason('arm', 'baseline_episode_count_insufficient', 'At least 100 complete baseline episodes are required; smoke 3/3 does not qualify.'))
    if record.get('smoke') is True or record.get('smoke_only') is True:
        reasons.append(_reason('arm', 'smoke_evidence_not_eligible', 'Smoke evidence cannot admit larger training.'))
    if not isinstance(trials, list) or type(episodes) is not int or len(trials) != episodes:
        reasons.append(_reason('arm', 'baseline_trial_records_missing', 'Full per-episode evidence must match the episode count.'))
        trials = []
    valid_trial_shape = bool(trials)
    seeds, wins, failures = [], 0, Counter()
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            valid_trial_shape = False
            continue
        if type(trial.get('seed')) is not int or type(trial.get('is_success')) is not bool:
            valid_trial_shape = False
        else:
            seeds.append(trial['seed'])
            wins += int(trial['is_success'])
        if ('failure_reason' not in trial or 'first_collision' not in trial
                or 'first_physics_anomaly' not in trial
                or not isinstance(trial.get('guard_events'), list)
                or not _finite(trial.get('q_tracking_error_rad'))):
            valid_trial_shape = False
        failure = trial.get('failure_reason')
        if failure is not None and not isinstance(failure, str):
            valid_trial_shape = False
        if isinstance(failure, str) and failure:
            failures[failure] += 1
        if trial.get('is_success') is True and failure:
            valid_trial_shape = False
    if len(set(seeds)) != len(seeds):
        reasons.append(_reason('arm', 'duplicate_baseline_seeds', 'Repeated episodes cannot inflate the sample count.'))
    if not valid_trial_shape:
        reasons.append(_reason('arm', 'baseline_guard_evidence_missing', 'Each v3 trial needs explicit guard_events, first_collision, first_physics_anomaly, finite tracking error and a Boolean result.'))
    denominator = episodes if type(episodes) is int and episodes > 0 else 1
    rate = wins / denominator
    if type(record.get('successes')) is not int or record.get('successes') != wins:
        reasons.append(_reason('arm', 'baseline_success_count_mismatch', 'Summary successes must equal complete per-trial results.'))
    if not _finite(record.get('success_rate')) or abs(record['success_rate'] - rate) > 1e-9:
        reasons.append(_reason('arm', 'baseline_success_rate_mismatch', 'Summary success rate must equal the full-episode denominator.'))
    if rate < .95:
        reasons.append(_reason('arm', 'baseline_success_below_gate', f'Observed baseline success {rate:.4f} is below 0.95.'))
    if not isinstance(record.get('failures'), dict) or record['failures'] != dict(failures):
        reasons.append(_reason('arm', 'baseline_failure_summary_mismatch', 'Failure summary must agree with full per-trial evidence.'))
    events = _serious_events({'failures': record.get('failures'), 'trials': trials}, 'baseline')
    if events:
        reasons.append(_reason('arm', 'serious_guard_events_present', 'Hard contact, collision, joint/closure limits, tracking faults or invalid physics forbid expansion even when success rate passes.'))
    if _contains_nonfinite(record):
        reasons.append(_reason('arm', 'nonfinite_evaluation_content', 'NaN/Infinity anywhere in evaluation evidence is rejected.'))
    captured = record.get('provenance')
    config = captured.get('config', {}) if isinstance(captured, dict) else {}
    if not isinstance(config, dict):
        config = {}
    reasons.extend(_threshold_check(config))
    return {'valid': not reasons, 'episodes': episodes, 'successes_from_trials': wins,
            'success_rate_from_trials': rate, 'minimum_episodes': 100, 'minimum_success_rate': .95,
            'serious_event_count': len(events), 'serious_events': events,
            'guard_evidence_complete': valid_trial_shape,
            'statistical_note': 'An observed-rate gate is not a guarantee that population success is at least 95%.'}, reasons


def _mobile_check(record):
    reasons = []
    episodes, trials = record.get('episodes'), record.get('trials')
    if type(episodes) is not int or episodes < 20:
        reasons.append(_reason('mobile', 'mobile_episode_count_insufficient', 'At least 20 complete mobile episodes are required.'))
    if record.get('learned_vision') is not True:
        reasons.append(_reason('mobile', 'learned_vision_not_verified', 'Contract/colour/mask fixtures cannot validate learned visual recognition.'))
    backend = record.get('vision_backend')
    if not isinstance(backend, str) or backend not in REAL_SEGMENTERS:
        reasons.append(_reason('mobile', 'real_segmenter_backend_missing', 'A documented real target-bowl segmenter backend is required; contract backend is ineligible.'))
    if not isinstance(trials, list) or type(episodes) is not int or len(trials) != episodes:
        reasons.append(_reason('mobile', 'mobile_trial_records_missing', 'Mobile metrics require the full episode denominator and per-episode records.'))
        trials = []
    keys = ('recognition_success', 'parking_success', 'precise_localization_success')
    counts, seeds = Counter(), []
    complete = bool(trials)
    for trial in trials:
        if not isinstance(trial, dict):
            complete = False
            continue
        if type(trial.get('seed')) is not int:
            complete = False
        else:
            seeds.append(trial['seed'])
        for key in keys:
            if type(trial.get(key)) is not bool:
                complete = False
            counts[key] += int(trial.get(key) is True)
        # A fixture cannot be hidden inside an otherwise learned-vision report.
        trial_backend = trial.get('vision_backend', backend)
        if (trial.get('learned_vision', True) is not True or not isinstance(trial_backend, str)
                or trial_backend not in REAL_SEGMENTERS):
            complete = False
    if not complete:
        reasons.append(_reason('mobile', 'mobile_metric_evidence_incomplete', 'Each episode needs seed and Boolean recognition, parking and precise-localization results from the real segmenter.'))
    if len(set(seeds)) != len(seeds):
        reasons.append(_reason('mobile', 'duplicate_mobile_seeds', 'Repeated mobile episodes cannot inflate the sample count.'))
    denominator = episodes if type(episodes) is int and episodes > 0 else 1
    rates = {key: counts[key] / denominator for key in keys}
    for key, rate in rates.items():
        if rate < .95:
            reasons.append(_reason('mobile', key + '_below_gate', f'{key} observed rate {rate:.4f} is below 0.95.'))
    # A complete platform pipeline must also complete the physical grasp.
    # Seeing the bowl alone is not evidence of pick-and-place reliability.
    if record.get('pipeline')=='observed_platform_drive_wrist_physical_grasp_v4':
        wins=sum(trial.get('is_success') is True for trial in trials)
        rates['full_pick_place']=wins/denominator
        if record.get('localization_only') is not False or any(type(t.get('is_success')) is not bool for t in trials):
            reasons.append(_reason('mobile','full_grasp_evidence_missing','Require full physical-grasp episodes, not localization-only.'))
        if rates['full_pick_place']<.95:
            reasons.append(_reason('mobile','full_pick_place_below_gate','Observed complete pipeline success must be at least 95%.'))
    events = _serious_events(record, 'mobile')
    if events:
        reasons.append(_reason('mobile', 'mobile_serious_events_present', 'Collision, tracking or physics anomalies block mobile learning expansion.'))
    if _contains_nonfinite(record):
        reasons.append(_reason('mobile', 'nonfinite_mobile_content', 'Nonfinite mobile evaluation values are rejected.'))
    return {'valid': not reasons, 'episodes': episodes, 'learned_vision': record.get('learned_vision') is True,
            'vision_backend': backend, 'metric_successes': dict(counts), 'metric_rates': rates,
            'minimum_episodes': 20, 'minimum_metric_rate': .95,
            'serious_event_count': len(events), 'serious_events': events,
            'planning_mode': 'deterministic_parking_and_forward_path_search'}, reasons


def build_training_gate(baseline, mobile, tests_log, *, expected_provenance=None):
    """Return fail-closed admission decisions; perform no training or evaluation."""
    reasons = []
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(mobile, dict):
        mobile = {}
    if not isinstance(tests_log, str):
        tests_log = ''
    expected = dict(expected_provenance or {})
    if expected_provenance is None:
        try:
            expected = current_package_provenance()
        except (OSError, ValueError) as error:
            reasons.append(_reason('shared', 'current_package_fingerprint_unavailable', str(error)))
    tests = _tests_check(tests_log)
    if not tests['valid']:
        reasons.append(_reason('shared', 'completed_tests_not_verified', 'Need a completed nonempty unittest log with an exact OK line and no FAILED/ERROR/Traceback or skipped-only completion.'))
    baseline_prov, issues = _provenance_check(baseline, 'arm', expected)
    reasons.extend(issues)
    mobile_prov, issues = _provenance_check(mobile, 'mobile', expected)
    reasons.extend(issues)
    bp, mp = baseline.get('provenance', {}), mobile.get('provenance', {})
    if isinstance(bp, dict) and isinstance(mp, dict):
        for key in ('package_generation', *HASH_KEYS):
            if not bp.get(key) or bp.get(key) != mp.get(key):
                reasons.append(_reason('mobile', 'baseline_mobile_provenance_mismatch', f'Baseline and mobile evidence differ or are missing {key}.'))
    arm, issues = _baseline_check(baseline)
    reasons.extend(issues)
    mobile_check, issues = _mobile_check(mobile)
    reasons.extend(issues)
    shared_ok = tests['valid'] and not any(item['scope'] == 'shared' for item in reasons)
    arm_allowed = shared_ok and baseline_prov['valid'] and arm['valid']
    mobile_allowed = (arm_allowed and mobile_prov['valid'] and mobile_check['valid']
                      and not any(item['scope'] == 'mobile' for item in reasons))
    actions = []
    if not arm_allowed:
        actions.append('Keep large PPO training disabled. Repair the reported model/contact/control fault or complete missing v3 baseline/tests, then rerun on the Mac with unchanged evidence thresholds.')
    else:
        actions.append('Allow bounded arm-residual simulation training: 20k steps first, compare fixed development scenarios and guards, then consider 100k only if checks continue to pass. Do not jump directly to 300k.')
    if not mobile_allowed:
        actions.append('Keep deterministic mobile planning and collect at least 20 full real-segmenter simulation episodes; report main recognition, collision-free parking and stopped wrist localization separately.')
    else:
        actions.append('Mobile evidence permits limited component experiments. Retain deterministic parking/navigation; end-to-end mobile-grasp policy training remains disabled in this package.')
    return _json_safe({'schema_version': 1, 'package_generation': GENERATION,
            'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'arm_residual_training_allowed': arm_allowed, 'mobile_learning_allowed': mobile_allowed,
            'end_to_end_policy_training_allowed': False,
            'training_schedule_if_allowed': [20000, 100000],
            'checks': {'tests': tests, 'baseline_provenance': baseline_prov,
                       'mobile_provenance': mobile_prov, 'baseline': arm, 'mobile': mobile_check},
            'expected_provenance': expected, 'reasons': reasons,
            'recommended_next_action': actions,
            'hardware_safety_certification': False, 'hardware_output_enabled': False,
            'scope': 'simulation_evidence_admission_only_not_hardware_or_deployment_certification',
            'config_scope': 'Captured configuration hashes must match. A training launcher must preserve and recheck that config hash before every stage.'})


def _read_json(path):
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('evaluation_file_exceeds_64MiB_limit')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('evaluation_must_be_a_JSON_object')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--mobile', type=Path, required=True)
    parser.add_argument('--tests-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-provenance', type=Path,
                        help='Optional JSON provenance from the same immutable training configuration; current source/model hashes are still checked.')
    args = parser.parse_args(argv)
    input_errors = []
    records = {}
    for name, path in [('baseline', args.baseline), ('mobile', args.mobile)]:
        try:
            records[name] = _read_json(path)
        except (OSError, ValueError) as error:
            records[name] = {}
            input_errors.append(_reason(name, 'input_unreadable_or_invalid', str(error)))
    try:
        log = args.tests_log.read_text()
    except (OSError, UnicodeError) as error:
        log = ''
        input_errors.append(_reason('shared', 'tests_log_unreadable', str(error)))
    expected = None
    if args.expected_provenance:
        try:
            expected = current_package_provenance()
            declared = _read_json(args.expected_provenance)
            declared = declared.get('provenance', declared)
            # This option narrows config expectations; it cannot override current
            # package source/model fingerprints with old user-supplied values.
            if not _hash(declared.get('config_sha256')):
                raise ValueError('expected_config_sha256_missing')
            expected['config_sha256'] = declared['config_sha256']
        except (OSError, ValueError, AttributeError) as error:
            input_errors.append(_reason('shared', 'expected_provenance_invalid', str(error)))
    report = build_training_gate(records['baseline'], records['mobile'], log, expected_provenance=expected)
    if input_errors:
        report['reasons'] = input_errors + report['reasons']
        if any(item['scope'] in ('baseline', 'arm', 'shared') for item in input_errors):
            report['arm_residual_training_allowed'] = False
            report['recommended_next_action'] = [
                'Do not start training. Repair missing/unreadable evidence or expected provenance and regenerate this gate report.',
                'Keep mobile parking deterministic; end-to-end policy training remains disabled.']
        report['mobile_learning_allowed'] = False
    report['inputs'] = {'baseline': str(args.baseline.resolve()), 'mobile': str(args.mobile.resolve()),
                        'tests_log': str(args.tests_log.resolve())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('arm_residual_training_allowed', 'mobile_learning_allowed',
                     'end_to_end_policy_training_allowed', 'recommended_next_action')}, ensure_ascii=False, indent=2))
    return 0 if report['arm_residual_training_allowed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
