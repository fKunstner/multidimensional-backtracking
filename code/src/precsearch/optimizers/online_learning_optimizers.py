from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy._typing import NDArray

from precsearch.problems import OptProblem, OptProblemAtPoint


def solve_adagrad(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    diagonal=True,
    starting_stepsize: float = 1e-2,
    D: float = 10**2,
    project: bool = False,
    use_linesearch: bool = False,
    maxiter: int = 1000,
    tol: float = 10**-3,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    """

    Args:
        func:
        grad:
        x0:
        starting_stepsize: default of 1e-2 from pytorch implementation
        D: radius of convex set (only used if project = True)
        project: whether to project
        maxiter:
        tol:
        callback:

    Returns:

    """

    def iteration(Sx: OptProblemAtPoint, state=None):
        if diagonal:
            state += Sx.g**2
        else:
            state += np.linalg.norm(Sx.g) ** 2

        xnext = x - (starting_stepsize / np.sqrt(state)) * Sx.g

        if project and np.linalg.norm(xnext) > D:
            xnext = D * xnext / np.linalg.norm(xnext)

        return Sx.at_y(xnext), state

    problem = OptProblem(func, grad)
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)
    nfev = 0
    ngev = 0
    nbacktrack = 0

    if callback is not None:
        callback(
            prob_at_x,
            {
                "stepsize": starting_stepsize,
                "nfev": nfev,
                "ngev": ngev,
                "nbacktrack": nbacktrack,
            },
        )

    state = 0
    for t in range(maxiter):
        prob_at_x, state = iteration(prob_at_x, state)
        nfev += 1
        ngev += 1

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "stepsize": starting_stepsize / np.sqrt(state),
                    "nfev": nfev,
                    "ngev": ngev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x


def solve_adagrad_ls(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    forward: float = 1.1,
    backward: float = 0.5,
    maxiter: int = 1000,
    tol: float = 10**-3,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    def armijo_gap(Sy, Sx, step):
        return Sy.f - Sx.f + 0.5 * np.inner(Sx.g, step)

    def precondition(grad, sgs):
        prec = np.zeros_like(sgs)
        prec[sgs > 0] = 1 / np.sqrt(sgs[sgs > 0])
        return np.multiply(prec, grad)

    def iteration(Sx: OptProblemAtPoint, _sum_grad_squared, _ss):
        step = _ss * precondition(Sx.g, _sum_grad_squared)
        Sy = Sx.at_y(Sx.x - step)
        if armijo_gap(Sy, Sx, step) < 0:
            return Sy, _sum_grad_squared + Sy.g**2, _ss * forward
        else:
            return Sx, _sum_grad_squared, _ss * backward

    problem = OptProblem(func, grad)
    prob_at_x = OptProblemAtPoint(problem, np.copy(x0))
    sum_grad_squared = prob_at_x.g**2

    def find_initial_stepsize(Sx):
        ss = 1.0
        step = precondition(Sx.g, np.sqrt(sum_grad_squared))
        nfev = 0

        Sy = Sx.at_y(Sx.x - ss * step)
        while armijo_gap(Sy, Sx, ss * step) < 0:
            ss = ss * forward
            Sy = Sx.at_y(Sx.x - ss * step)
            nfev += 1

        while armijo_gap(Sy, Sx, ss * step) > 0:
            ss = ss * backward
            Sy = Sx.at_y(Sx.x - ss * step)
            nfev += 1

        return ss, nfev

    ss, nfev = find_initial_stepsize(prob_at_x)
    ngev = 0
    nbacktrack = 0

    if callback is not None:
        callback(
            prob_at_x,
            {
                "stepsize": ss,
                "stepsizes": ss
                * precondition(np.ones_like(sum_grad_squared), sum_grad_squared),
                "nfev": nfev,
                "ngev": ngev,
                "nbacktrack": nbacktrack,
            },
        )

    for t in range(maxiter):
        new_prob_at_x, sum_grad_squared, ss = iteration(prob_at_x, sum_grad_squared, ss)

        iteration_type = "backtrack" if new_prob_at_x is prob_at_x else "update"
        prob_at_x = new_prob_at_x

        if iteration_type == "backtrack":
            nfev += 1
            nbacktrack += 1
        elif iteration_type == "update":
            nfev += 1
            ngev += 1

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "stepsize": ss,
                    "stepsizes": ss
                    * precondition(np.ones_like(sum_grad_squared), sum_grad_squared),
                    "nfev": nfev,
                    "ngev": ngev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x
