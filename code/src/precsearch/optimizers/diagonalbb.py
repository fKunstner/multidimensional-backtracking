from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy._typing import NDArray

from precsearch.problems import OptProblem, OptProblemAtPoint


def solve_diagonalbb(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    starting_stepsize: float = 10**-6,
    mu: float = 10**-6,
    backtrack: float = 0.5,
    ls_window: int = 15,
    maxiter: int = 1000,
    tol: float = 10**-4,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    def progress_gap(new_f: float, f: float, P: NDArray, g: NDArray) -> float:
        return new_f - f + 0.5 * np.inner(g, P * g)

    def sufficient_nonmonotonic_progress(
        new_f: float, f_hist: list, P: NDArray, g: NDArray
    ) -> bool:
        f_max = np.max(f_hist)
        return progress_gap(new_f, f_max, P, g) <= 0

    def diagonal_bb_prec(
        x: NDArray, x_prev: NDArray, g: NDArray, g_prev: NDArray, U: NDArray
    ):
        s = x - x_prev
        y = g - g_prev
        d = len(g)

        newU = (s * y + mu * U) / (s * s + mu * np.ones(d))
        condition_1 = condition_2 = np.ones(d) < 0

        if np.dot(s, y) != 0:
            stepsize_bb1 = (np.linalg.norm(s) ** 2) / np.dot(s, y)
            stepsize_bb2 = np.dot(s, y) / (np.linalg.norm(y) ** 2)
            condition_1 = np.ones(d) / stepsize_bb1 > (s * y + mu * U) / (
                s**2 + mu * np.ones(d)
            )

            condition_2 = np.ones(d) / stepsize_bb2 < (s * y + mu * U) / (
                s**2 + mu * np.ones(d)
            )

            newU[condition_1] = 1 / stepsize_bb1
            newU[condition_2] = 1 / stepsize_bb2

        return newU

    def iteration(Sx: OptProblemAtPoint, U: NDArray, f_values_window: list):
        Sy = Sx.at_y(Sx.x - (1 / U) * Sx.g)

        if sufficient_nonmonotonic_progress(Sy.f, f_values_window, 1 / U, Sx.g):
            newU = diagonal_bb_prec(Sy.x, Sx.x, Sy.g, Sx.g, U)
            new_f_values_window = f_values_window[1:]
            new_f_values_window.append(Sy.f)
            return Sy, newU, new_f_values_window
        else:
            return Sx, (1 / backtrack) * U, f_values_window

    dim = len(x0)
    problem = OptProblem(func, grad)
    x = np.copy(x0)
    prob_at_x = OptProblemAtPoint(problem, x)
    U = diagonal_bb_prec(
        prob_at_x.x,
        prob_at_x.x,
        prob_at_x.g,
        prob_at_x.g,
        starting_stepsize * np.ones(dim),
    )
    f_values_window = ls_window * [-np.inf]
    pos_next_f_val = 0

    f_values_window = f_values_window[1:]
    f_values_window.append(prob_at_x.f)

    nfev = 0
    ngev = 0
    nbacktrack = 0

    if callback is not None:
        stepsizes = None
        callback(
            prob_at_x,
            {
                "prec": U,
                "nfev": nfev,
                "ngev": ngev,
                "nbacktrack": nbacktrack,
            },
        )

    for t in range(maxiter):
        new_prob_at_x, U, f_values_window = iteration(prob_at_x, U, f_values_window)

        iteration_type = "backtrack" if new_prob_at_x is prob_at_x else "update"

        prob_at_x = new_prob_at_x

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if iteration_type == "backtrack":
            nfev += 1
            nbacktrack += 1
        elif iteration_type == "update":
            nfev += 1
            ngev += 1

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "prec": U,
                    "nfev": nfev,
                    "ngev": ngev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x
