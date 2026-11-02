"""Bounded image-space correction using measured joint responses (px/P)."""
import numpy as np
from scipy.optimize import lsq_linear


def allocate_correction(jacobian, error_px, lower, upper, gain=.7,
                        damping=.08, weights=None):
    """Minimise image error with joint limits inside the optimisation.

    Values are actual joint displacements, before directional servo offsets.
    A fixed joint is removed before solving; remaining joints take its task.
    """
    matrix = np.asarray(jacobian, dtype=float)
    error = np.asarray(error_px, dtype=float)
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != 2 or error.shape != (2,)
            or low.shape != (matrix.shape[1],) or high.shape != low.shape
            or not all(np.isfinite(a).all() for a in (matrix, error, low, high))
            or np.any(low > 0) or np.any(high < 0) or np.any(low > high)):
        raise ValueError('Invalid image response or displacement bounds')
    penalty = np.ones(len(low)) if weights is None else np.asarray(weights, dtype=float)
    if penalty.shape != low.shape or not np.isfinite(penalty).all() or np.any(penalty <= 0):
        raise ValueError('Joint weights must be finite and positive')
    if not 0 < gain <= 1 or not 0 < damping <= 1:
        raise ValueError('Invalid correction gain or damping')
    free = (high-low > 1e-6) & (np.linalg.norm(matrix, axis=0) > .01)
    result = np.zeros(len(low))
    if free.any():
        n = int(free.sum())
        augmented = np.vstack((matrix[:, free], damping*np.diag(penalty[free])))
        target = np.concatenate((-gain*error, np.zeros(n)))
        solution = lsq_linear(augmented, target, bounds=(low[free], high[free]),
                              method='bvls', tol=1e-8, max_iter=50)
        if not solution.success or not np.isfinite(solution.x).all():
            raise ValueError('Joint allocation did not converge')
        result[free] = solution.x
    return result
