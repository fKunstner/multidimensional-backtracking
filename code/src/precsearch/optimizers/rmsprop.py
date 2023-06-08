from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy._typing import NDArray

from precsearch.problems import OptProblem, OptProblemAtPoint


def solve_rmsprop(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    starting_stepsize: float = 10**-6,
    avg_decay: float = 0.9,
    denominator_offset: float = 10**-6,
    backtrack: float = 0.5,
    maxiter: int = 1000,
    tol: float = 10**-4,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    def iteration(Sx: OptProblemAtPoint, g2_avg: NDArray):
        normalization_vec = np.sqrt(g2_avg + denominator_offset)

        Sy = Sx.at_y(Sx.x - stepsize * (1 / normalization_vec) * Sx.g)

        g2_avg = avg_decay * g2_avg + (1 - avg_decay) * Sy.g**2
        return Sy, g2_avg

    problem = OptProblem(func, grad)
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)
    stepsize = starting_stepsize

    moving_gsquared_avg = (1 - avg_decay) * prob_at_x.g**2

    nfev = 0
    ngev = 0
    nbacktrack = 0

    # Find good starting step-size with LS

    def armijo(Sy, Sx, alpha, P):
        return Sy.f - Sx.f + alpha * 0.5 * np.inner(Sx.g, P * Sx.g)

    Sx = prob_at_x
    Sy = Sx.at_y(Sx.x - stepsize * Sx.g)
    normalization_vec = np.sqrt(moving_gsquared_avg + denominator_offset)
    P = 1 / normalization_vec
    while armijo(Sy, Sx, stepsize, P) < 0:
        stepsize /= backtrack
        Sy = Sx.at_y(Sx.x - stepsize * P * Sx.g)
        nfev += 1
        nbacktrack += 1
    while armijo(Sy, Sx, stepsize, P) > 0:
        stepsize *= backtrack
        Sy = Sx.at_y(Sx.x - stepsize * P * Sx.g)
        nfev += 1
        nbacktrack += 1

    if callback is not None:
        callback(
            prob_at_x,
            {
                "nfev": nfev,
                "ngev": ngev,
                "nbacktrack": nbacktrack,
            },
        )

    for t in range(maxiter):
        prob_at_x, moving_gsquared_avg = iteration(prob_at_x, moving_gsquared_avg)

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        ngev += 1

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "nfev": nfev,
                    "ngev": ngev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x
