from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy._typing import NDArray

from precsearch.problems import OptProblem, OptProblemAtPoint


def solve_hypergd(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    starting_stepsize: float = 10**-2,
    hyper_stepsize: float = 10**-4,
    multiplicative_update: bool = False,
    backtrack: float = 0.5,
    maxiter: int = 1000,
    tol: float = 10**-3,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    def iteration(Sx: OptProblemAtPoint, stepsize: float, prev_grad: NDArray):
        hyper_grad = -np.dot(Sx.g, prev_grad)
        new_stepsize = stepsize

        if not multiplicative_update:
            new_stepsize = stepsize - hyper_stepsize * hyper_grad
        else:
            normalizing_cst = np.linalg.norm(Sx.g) * np.linalg.norm(prev_grad)
            if normalizing_cst > 0:
                new_stepsize = stepsize * (
                    1 - hyper_stepsize * hyper_grad / normalizing_cst
                )

        Sy = Sx.at_y(Sx.x - new_stepsize * Sx.g)

        return Sy, new_stepsize, Sx.g

    dim = len(x0)
    problem = OptProblem(func, grad)
    prev_grad = np.zeros(dim)
    stepsize = starting_stepsize
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)

    def armijo(Sy, Sx, alpha):
        return Sy.f - Sx.f + alpha * 0.5 * np.inner(Sx.g, Sx.g)
    
    # Find good starting step-size with LS
    Sx = prob_at_x
    Sy = Sx.at_y(Sx.x - stepsize * Sx.g)
    while armijo(Sy, Sx, stepsize) < 0:
        stepsize /= backtrack
        Sy = Sx.at_y(Sx.x - stepsize * Sx.g)
    while armijo(Sy, Sx, stepsize) > 0:
        stepsize *= backtrack
        Sy = Sx.at_y(Sx.x - stepsize * Sx.g)



    if callback is not None:
        callback(
            prob_at_x,
            {
                "stepsize": stepsize,
            },
        )

    for t in range(maxiter):
        prob_at_x, stepsize, prev_grad = iteration(prob_at_x, stepsize, prev_grad)

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "stepsize": stepsize,
                },
            )

    return prob_at_x
