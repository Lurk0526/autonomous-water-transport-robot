"""Controller primitives; no hardware output is generated."""
import numpy as np


def synchronized_step(current, target, max_delta):
    """Scale every joint by the same factor, preserving coordinated joint motion."""
    delta=np.asarray(target)-np.asarray(current)
    return np.asarray(current)+delta*min(1.,max_delta/max(float(np.max(np.abs(delta))),1e-12))


def smoothstep(u):
    u=float(np.clip(u,0.,1.))
    return u**3*(10+u*(-15+6*u))


def checkpoint_key(result):
    """Rank success first; prefer fewer collisions on a tie, then smaller placement error."""
    return (result['successes'], -result['failures'].get('unexpected_collision',0),
            -float(np.mean([r['place_error_m'] for r in result['trials']])))
