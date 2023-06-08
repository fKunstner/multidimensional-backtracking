from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy._typing import NDArray

from precsearch.problems import OptProblem, OptProblemAtPoint


def solve_rprop(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    starting_stepsize: float = 10**-2,
    eta_plus: float = 1.2,
    eta_minus: float = 0.5,
    max_stepsize: float = 50,
    min_stepsize: float = 10**-6,
    maxiter: int = 1000,
    tol: float = 10**-3,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    def iteration(Sx: OptProblemAtPoint, stepsize_vec: NDArray, prev_grad: NDArray):
        new_stepsize_vec = np.copy(stepsize_vec)
        new_grad = np.copy(Sx.g)

        grad_change = prev_grad * new_grad

        new_stepsize_vec[grad_change > 0] = stepsize_vec[grad_change > 0] * eta_plus
        new_stepsize_vec[grad_change < 0] = stepsize_vec[grad_change < 0] * eta_minus
        new_grad[grad_change < 0] = 0
        new_stepsize_vec = new_stepsize_vec.clip(min_stepsize, max_stepsize)

        Sy = Sx.at_y(Sx.x - new_stepsize_vec * np.sign(new_grad))

        return Sy, new_stepsize_vec, new_grad

    dim = len(x0)
    problem = OptProblem(func, grad)
    prev_grad = np.zeros(dim)
    stepsize_vec = starting_stepsize * np.ones(dim)
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)

    if callback is not None:
        stepsizes = None
        callback(
            prob_at_x,
            {
                "stepsizes": stepsize_vec,
            },
        )

    for t in range(maxiter):
        new_prob_at_x, stepsize_vec, prev_grad = iteration(
            prob_at_x, stepsize_vec, prev_grad
        )

        prob_at_x = new_prob_at_x

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "stepsizes": stepsize_vec,
                },
            )

    return prob_at_x
