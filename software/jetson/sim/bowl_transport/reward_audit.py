"""CPU-only mathematical reward audit; NOT an RL task, physics, or spill estimator.

All defaults are uncalibrated candidates. The synthetic water masses below are
test fixtures, not measurements of the user's bowl. Run this file to audit attacks.
The interface returns a per-transition reward already integrated over dt. When
adapting to Isaac Lab's RewardManager (which also multiplies terms by dt), divide
this complete transition reward by dt, or use a Direct environment without that
extra multiplication. Event and shaping rewards must not be multiplied twice.
"""
from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path


def finite(value, name, minimum=0.0, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} outside allowed bounds')
    return float(value)


@dataclass(frozen=True)
class RewardConfig:
    # A fixed task deadline is part of the task; expose remaining time to policy.
    horizon_s: float = 120.0
    discount_at_horizon: float = 0.95
    potential_bound: float = 0.10
    soft_budget: float = 0.20
    loss_budget: float = 0.25
    success_bonus: float = 1.0
    failure_penalty: float = 2.0

    def __post_init__(self):
        finite(self.horizon_s, 'horizon', 1e-6)
        finite(self.discount_at_horizon, 'discount', 1e-6, 1.0)
        for name in ('potential_bound', 'soft_budget', 'loss_budget',
                     'success_bonus', 'failure_penalty'):
            finite(getattr(self, name), name)
        # Bound every valid successful episode above zero and every failed
        # episode below zero, regardless of bounded shaping/soft costs.
        if (self.discount_at_horizon * self.success_bonus
                <= self.soft_budget + self.loss_budget + self.potential_bound):
            raise ValueError('success separation is not guaranteed')
        if self.failure_penalty <= 0:
            raise ValueError('failure penalty must be positive')


class RewardAudit:
    """Stateful finite-horizon accounting with irrevocable, once-only termination.

    soft_costs are dimensionless [0,1] densities: motion, risk_proxy, intervention.
    Time consumes one fourth of the soft budget; the other three share the rest.
    risk_proxy is only a risk proxy, NEVER grams of water. progress in [0,1] is a
    bounded potential, not a repeated positive reward for reaching a phase.
    cumulative_loss_g must be irreversible escaped-water mass, not instantaneous
    water missing from the bowl. Unknown loss remains unknown for the episode.
    Outcome evidence must come from an independent supervisor/measurement layer.
    """
    def __init__(self, initial_water_g=None, config=None, initial_progress=0.0,
                 max_spill_g=None):
        self.cfg = config or RewardConfig()
        if initial_water_g is not None:
            finite(initial_water_g, 'initial_water_g', 1e-9)
        self.initial_water_g = initial_water_g
        if max_spill_g is not None:
            finite(max_spill_g, 'max_spill_g', 0, initial_water_g)
        self.max_spill_g = max_spill_g
        self.t = 0.0
        self.potential = finite(initial_progress, 'progress', 0, 1) * self.cfg.potential_bound
        self.previous_loss_g = 0.0
        self.complete_evidence = initial_water_g is not None
        self.discounted_known_component = 0.0
        self.terminal = None

    def advance(self, dt, *, progress=0.0, motion=0.0, risk_proxy=0.0,
                intervention=0.0, cumulative_loss_g=None, outcome=None):
        if self.terminal is not None:
            raise RuntimeError('episode is terminal; rewards cannot be repeated')
        dt = finite(dt, 'dt', 1e-12)
        remaining = self.cfg.horizon_s - self.t
        if dt > remaining + 1e-8:
            raise ValueError('step crosses fixed deadline; split the last step')
        dt = min(dt, remaining)
        progress = finite(progress, 'progress', 0, 1)
        costs = [finite(v, n, 0, 1) for n, v in
                 [('motion', motion), ('risk_proxy', risk_proxy), ('intervention', intervention)]]
        if outcome not in (None, 'success', 'failure'):
            raise ValueError('outcome must be success, failure, or None')
        if cumulative_loss_g is not None:
            finite(cumulative_loss_g, 'cumulative_loss_g')
            if cumulative_loss_g < self.previous_loss_g:
                raise ValueError('irreversible loss cannot decrease')
            if self.initial_water_g is not None and cumulative_loss_g > self.initial_water_g:
                raise ValueError('escaped water exceeds initial water')
        terminal = outcome
        if outcome == 'success' and (self.max_spill_g is None or
                cumulative_loss_g is None or not self.complete_evidence):
            terminal = 'unverified_success'
            self.complete_evidence = False
        if (self.max_spill_g is not None and cumulative_loss_g is not None
                and cumulative_loss_g > self.max_spill_g):
            terminal = 'failure'  # A success event cannot override this constraint.
        if self.t + dt >= self.cfg.horizon_s - 1e-8 and terminal is None:
            terminal = 'timeout'
        rate = -math.log(self.cfg.discount_at_horizon) / self.cfg.horizon_s
        gamma = math.exp(-rate * dt)
        integral_dt = -math.expm1(-rate * dt) / rate if rate else dt
        density = self.cfg.soft_budget / self.cfg.horizon_s * (1.0 + sum(costs)) / 4.0
        soft = -density * integral_dt
        next_potential = 0.0 if terminal else self.cfg.potential_bound * progress
        shaping = gamma * next_potential - self.potential
        # Failure has fixed episode cost: delaying it must not discount it away.
        discount_so_far = math.exp(-rate * self.t)
        event = (gamma * self.cfg.success_bonus if terminal == 'success' else
                 -self.cfg.failure_penalty / discount_so_far if terminal else 0.0)
        # Loss increments receive fixed episode cost regardless of when sampled.
        # This does not remove physical uncertainty in the spill measurement.
        loss = None
        if cumulative_loss_g is not None and self.initial_water_g is not None:
            delta_loss = cumulative_loss_g - self.previous_loss_g
            # Fixed physical loss charge: postponing the same loss buys no reward.
            loss = -self.cfg.loss_budget * delta_loss / self.initial_water_g / discount_so_far
        self.complete_evidence = self.complete_evidence and loss is not None
        known = soft + shaping + event + (loss if loss is not None else 0.0)
        self.discounted_known_component += math.exp(-rate * self.t) * known
        self.t += dt
        self.potential = next_potential
        if cumulative_loss_g is not None:
            self.previous_loss_g = cumulative_loss_g
        self.terminal = terminal
        return {
            'candidate_reward': known if self.complete_evidence else None,
            'candidate_discounted_return': self.discounted_known_component if self.complete_evidence else None,
            'known_components_only_not_complete_reward': known,
            'loss_charge': loss,
            'soft': soft, 'shaping': shaping, 'event': event, 'gamma_dt': gamma,
            'terminal': terminal,
            'terminated': terminal is not None,
            'truncated': False,  # A task deadline is failure, not external truncation.
            'evidence_complete': self.complete_evidence,
        }


@dataclass(frozen=True)
class EpisodeMetrics:
    # scenario_id must include route, terrain, platform, bowl, start pose and seed.
    scenario_id: str
    source: str  # synthetic / physical_simulation / measured; never mix these.
    initial_water_g: float | None
    empty_bowl_g: float | None
    filled_total_g: float
    fill_condition: str
    outcome: str
    escaped_water_g: float | None
    duration_s: float
    hard_failure: bool = False


def compare_episodes(a, b, *, max_spill_g=None):
    """Independent lexicographic evaluation, never a weighted-reward leaderboard.

    Requires matched initial conditions and comparable evidence. A lower returned
    key is better. Unknown mass/loss refuses comparison rather than assuming zero.
    This function compares two matched runs, not statistical superiority.
    """
    if max_spill_g is not None:
        finite(max_spill_g, 'max_spill_g')
    for item in (a, b):
        if not item.scenario_id or item.source not in ('synthetic', 'physical_simulation', 'measured'):
            raise ValueError('scenario and evidence source must be explicit')
        if item.initial_water_g is None or item.empty_bowl_g is None or item.escaped_water_g is None:
            raise ValueError('unknown water/bowl/spill mass: comparison blocked')
        finite(item.initial_water_g, 'initial water', 1e-9)
        finite(item.empty_bowl_g, 'empty bowl', 1e-9)
        finite(item.filled_total_g, 'filled total', 1e-9)
        finite(item.escaped_water_g, 'escaped water', 0, item.initial_water_g)
        finite(item.duration_s, 'duration', 0)
        if not math.isclose(item.initial_water_g + item.empty_bowl_g, item.filled_total_g,
                            rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError('inconsistent mass accounting')
        if item.outcome not in ('success', 'failure', 'timeout'):
            raise ValueError('unknown outcome')
    fields = ('scenario_id', 'source', 'initial_water_g', 'empty_bowl_g',
              'filled_total_g', 'fill_condition')
    if any(getattr(a, field) != getattr(b, field) for field in fields):
        raise ValueError('unmatched scenario/source/water/mass/fill: comparison blocked')
    def key(m):
        disqualified = (m.hard_failure or m.outcome != 'success'
                        or (max_spill_g is not None and m.escaped_water_g > max_spill_g))
        return (int(disqualified), m.escaped_water_g, m.duration_s)
    ka, kb = key(a), key(b)
    return {'a_key': ka, 'b_key': kb, 'preferred': 'a' if ka < kb else 'b' if kb < ka else 'tie',
            'spill_acceptance': 'threshold_applied' if max_spill_g is not None else 'undetermined_no_threshold',
            'scope': 'matched_pair_only_not_statistical_or_physical_validation'}


def run_attacks():
    """Deterministic adversarial accounting fixtures, not robot trajectories."""
    cfg = RewardConfig(horizon_s=10)

    def rollout(seconds, *, cycle=False, loss_g=0.0, outcome='success', dt=.1):
        audit = RewardAudit(300.0, cfg, max_spill_g=5)
        last = None
        for i in range(round(seconds / dt)):
            final = i == round(seconds / dt) - 1
            last = audit.advance(dt, progress=float(i % 2) if cycle else .5,
                                 cumulative_loss_g=loss_g if final else 0.0,
                                 outcome=outcome if final else None)
        return audit, last

    fast, fast_result = rollout(4)
    slow, slow_result = rollout(8)
    cycle, cycle_result = rollout(4, cycle=True)
    failed, failure_result = rollout(.1, outcome='failure')
    _, delayed_failure_result = rollout(8, outcome='failure')
    idle, idle_result = rollout(10, outcome=None)
    damaged, damaged_result = rollout(4, loss_g=100)
    unknown = RewardAudit(None, cfg).advance(.1, outcome='success')
    violated = RewardAudit(300, cfg, max_spill_g=5).advance(
        .1, cumulative_loss_g=6, outcome='success')
    repeated_rejected = False
    try:
        fast.advance(.1, outcome='success', cumulative_loss_g=0)
    except RuntimeError:
        repeated_rejected = True
    base = dict(scenario_id='synthetic_seed_17', source='synthetic', initial_water_g=300.,
                empty_bowl_g=350., filled_total_g=650., fill_condition='to_rim',
                outcome='success', escaped_water_g=0., duration_s=4.)
    mismatched_rejected = False
    try:
        compare_episodes(EpisodeMetrics(**base), EpisodeMetrics(**dict(base, initial_water_g=200., filled_total_g=550., fill_condition='low')))
    except ValueError:
        mismatched_rejected = True
    checks = {
        'failure_cannot_beat_success': failure_result['candidate_discounted_return'] < fast_result['candidate_discounted_return'],
        'delaying_failure_does_not_improve_return': delayed_failure_result['candidate_discounted_return'] < failure_result['candidate_discounted_return'],
        'idle_timeout_fails': idle_result['terminal'] == 'timeout' and idle_result['candidate_discounted_return'] < 0,
        'waiting_without_benefit_loses': slow_result['candidate_discounted_return'] < fast_result['candidate_discounted_return'],
        'phase_cycles_do_not_farm_shaping': math.isclose(cycle_result['candidate_discounted_return'], fast_result['candidate_discounted_return'], abs_tol=1e-12),
        'lost_water_never_improves_return': damaged_result['candidate_discounted_return'] < fast_result['candidate_discounted_return'],
        'repeat_terminal_reward_rejected': repeated_rejected,
        'unknown_spill_is_not_zero': unknown['candidate_reward'] is None and unknown['loss_charge'] is None,
        'lower_fill_comparison_rejected': mismatched_rejected,
        'spill_constraint_overrides_success_event': violated['terminal'] == 'failure' and violated['candidate_discounted_return'] < 0,
        'missing_acceptance_limit_blocks_success': RewardAudit(300, cfg).advance(.1,cumulative_loss_g=0,outcome='success')['candidate_reward'] is None,
    }
    return {
        'kind': 'candidate_reward_math_audit_not_training_or_physics',
        'calibration_status': 'uncalibrated_candidates',
        'all_passed': all(checks.values()), 'checks': checks,
        'synthetic_fixture_only': {'initial_water_g': 300, 'empty_bowl_g': 350, 'filled_total_g': 650},
        'returns': {name: result['candidate_discounted_return'] for name, result in
                    [('success_fast', fast_result), ('success_slow', slow_result),
                     ('success_with_phase_cycles', cycle_result), ('failure', failure_result),
                     ('idle_timeout', idle_result), ('success_with_loss', damaged_result), ('spill_unknown', unknown)]},
        'limits': ['No simulation, training, or measured spill performance is certified.',
                   'Weighted reward cannot establish water-loss acceptance; use matched metrics and explicit constraints.',
                   'PBRS equivalence requires this discount and zero terminal potential; do not clip or keep only positive shaping.',
                   'An external rollout truncation must bootstrap the final pre-reset state; it is not a task timeout.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = run_attacks()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n', encoding='utf8')
    print(text)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
