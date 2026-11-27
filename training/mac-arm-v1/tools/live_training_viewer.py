"""Preview the newest saved policy without changing the training process."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
torch.set_num_threads(1)
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from arm_training.env import BowlEnv
from arm_training.model import config


def newest(run):
    if (run / 'final.zip').exists():
        return run / 'final.zip'
    candidates = [(int(re.fullmatch(r'checkpoint_(\d+)_steps.zip', p.name)[1]), p)
                  for p in run.glob('checkpoint_*_steps.zip')]
    return max(candidates)[1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    env = BowlEnv(config(run / 'config-used.json'), visual=True)
    state = {'paused': False, 'reset': False, 'quit': False}
    def key(k):
        if k == 32: state['paused'] = not state['paused']
        if k == 82: state['reset'] = True
        if k == 81: state['quit'] = True
    loaded = None
    policy = None
    preview_episode = 0
    last_result = 'Not completed'
    def reset():
        nonlocal loaded, policy, preview_episode
        candidate = newest(run)
        if candidate and candidate != loaded:
            try:
                next_policy = PPO.load(candidate, device='cpu')
            except (OSError, ValueError, RuntimeError) as exc:
                print('Checkpoint not ready; retrying next episode:', exc, flush=True)
            else:
                policy, loaded = next_policy, candidate
                print('PREVIEW_LOADED', loaded.name, 'steps', policy.num_timesteps, flush=True)
        if policy is None:
            raise RuntimeError('No readable checkpoint is available')
        preview_episode += 1
        return env.reset(seed=3000000 + preview_episode)[0]
    obs = reset()
    done = False
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key,
                                     show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [.15, 0, .45]
            viewer.cam.distance = 1.1
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 135
        print('LIVE_PREVIEW_READY: separate real-time rollouts; newest weights load each episode.', flush=True)
        last_step = time.perf_counter()
        while viewer.is_running() and not state['quit']:
            now = time.perf_counter()
            if state['reset'] or (done and now - last_step > 1.2):
                with viewer.lock():
                    obs = reset()
                state['reset'] = False
                done = False
            if not state['paused'] and not done and now - last_step >= env.dt:
                action = policy.predict(obs, deterministic=True)[0]
                with viewer.lock():
                    obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_step = now
                if done:
                    last_result = 'SUCCESS' if info['is_success'] else str(info['failure_reason'])
                    print('PREVIEW_EPISODE', preview_episode, last_result, flush=True)
                    (ROOT / 'outputs/mac-live-preview-status.json').write_text(json.dumps({
                        'weights': str(loaded), 'policy_steps': policy.num_timesteps,
                        'preview_episode': preview_episode, 'last_result': last_result,
                        'is_training_worker': False, 'included_in_evaluation': False,
                    }, indent=2) + '\n')
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                'TRAINING PREVIEW\nWeights\nPreview episode / phase\nPrevious result\nControls',
                'Separate rollout; auto-refresh each episode\n' + loaded.name +
                f'\n{preview_episode} / {env.phase}\n{last_result}\nSpace pause | R restart | Q exit'))
            viewer.sync()
            time.sleep(.005)


if __name__ == '__main__':
    main()
