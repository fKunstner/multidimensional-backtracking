from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator

from precsearch.optimizers.lgfbs_scipy import fmin_l_bfgs_b
from precsearch.problems import OptProblem, OptProblemAtPoint


def solve_preconditioned_gd_ls(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    starting_stepsize: float = 1.0,
    backtrack: float = 0.5,
    forward: float = 2.0,
    preconditioner: Optional[LinearOperator] = None,
    maxiter: int = 1000,
    tol: float = 10**-3,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    P = preconditioner

    def iteration(Sx: OptProblemAtPoint, max_stepsize: float):
        def armijo(Sy, Sx, alpha):
            if P is None:
                return Sy.f - Sx.f + alpha * 0.5 * np.inner(Sx.g, Sx.g)
            else:
                return Sy.f - Sx.f + alpha * 0.5 * np.inner(Sx.g, P.matvec(Sx.g))

        alpha = max_stepsize / 2
        if P is None:
            Sy = Sx.at_y(Sx.x - alpha * Sx.g)
        else:
            Sy = Sx.at_y(Sx.x - alpha * P.matvec(Sx.g))

        if armijo(Sy, Sx, alpha) < 0:
            return Sy, max_stepsize * forward
        else:
            return Sx, max_stepsize * backtrack

    problem = OptProblem(func, grad)
    ss = starting_stepsize
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)
    nfev = 0
    ngev = 0
    nbacktrack = 0

    if callback is not None:
        stepsizes = None if P is None else ss * P.matvec(np.ones_like(prob_at_x.g))
        callback(
            prob_at_x,
            {
                "ss": ss,
                "stepsizes": stepsizes,
                "nfev": nfev,
                "ngev": ngev,
                "nbacktrack": nbacktrack,
            },
        )

    for t in range(maxiter):
        new_prob_at_x, new_ss = iteration(prob_at_x, ss)

        iteration_type = "backtrack" if new_prob_at_x is prob_at_x else "update"

        prob_at_x, ss = new_prob_at_x, new_ss

        if iteration_type == "backtrack":
            nfev += 1
            nbacktrack += 1
        elif iteration_type == "update":
            nfev += 1
            ngev += 1

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            stepsizes = None if P is None else ss * P.matvec(np.ones_like(prob_at_x.g))
            callback(
                prob_at_x,
                {
                    "ss": ss,
                    "stepsizes": stepsizes,
                    "nfev": nfev,
                    "ngev": ngev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x


def solve_lbfgs(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    L=10,
    iprint=0,
    tol: float = 10**-3,
    maxiter: int = 1000,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    problem = OptProblem(func, grad)
    x = np.copy(x0)

    def lbfgs_callback(x, state):
        return callback(OptProblemAtPoint(problem, x), state)

    callback(OptProblemAtPoint(problem, x), {})

    res = fmin_l_bfgs_b(
        func=func,
        x0=x,
        fprime=grad,
        factr=tol / np.finfo(float).eps,
        pgtol=tol,
        epsilon=tol,
        maxiter=maxiter,
        maxfun=20 * maxiter,
        m=L,
        callback=lbfgs_callback,
        iprint=iprint,
    )

    return OptProblemAtPoint(problem, res[0])


def solve_diag_h(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    diag_h: Callable[[NDArray], NDArray],
    x0: NDArray,
    stepsize: float,
    tol: float = 10**-3,
    maxiter: int = 1000,
    backward: float = 0.5,
    forward: float = 1.1,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    def iteration(Sx: OptProblemAtPoint, stepsize):
        P = 1 / Sx.diag_h

        step = stepsize * np.multiply(P, Sx.g)
        Sy = Sx.at_y(Sx.x - step)

        if Sy.f < Sx.f - 0.5 * np.inner(Sx.g, step):
            return Sy, stepsize * forward
        else:
            return Sx, stepsize * backward

    problem = OptProblem(func, grad, diag_hess=diag_h)
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)

    nfev = 0
    ngev = 0
    ndhev = 0
    nbacktrack = 0

    if callback is not None:
        callback(
            prob_at_x,
            {
                "ss": stepsize,
                "nfev": nfev,
                "ngev": ngev,
                "ndhev": ndhev,
                "nbacktrack": nbacktrack,
            },
        )

    for t in range(maxiter):
        new_prob_at_x, new_ss = iteration(prob_at_x, stepsize)

        iteration_type = "backtrack" if new_prob_at_x is prob_at_x else "update"

        prob_at_x, stepsize = new_prob_at_x, new_ss

        if iteration_type == "backtrack":
            nfev += 1
            nbacktrack += 1
        elif iteration_type == "update":
            nfev += 1
            ngev += 1
            ndhev += 1

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "ss": stepsize,
                    "nfev": nfev,
                    "ngev": ngev,
                    "ndhev": ndhev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x
