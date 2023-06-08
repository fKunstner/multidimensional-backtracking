import warnings
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
import scipy as sp
from numpy.typing import NDArray

from precsearch.problems import OptProblem, OptProblemAtPoint

##
# Constraint sets


class PreconditionerSet:
    pass

    def __init__(self, dim: int, scale: float, incr=None):
        """
        Create a new preconditioner set

        Args:
            dim: An integer representing the dimensionality of the feature space.
            scale: A float representing the scale of the initial set.
            incr: A float representing the factor by which to increase the box size when
                `increase()` is called. If not provided, a default value is calculated based
                on the dimensionality.
        """
        pass

    def get_tentative_preconditioner(self, grad: NDArray = None) -> NDArray:
        raise NotImplementedError()

    def cut(self, u: NDArray, refine_steps: int = 0, P=None) -> "PreconditionerSet":
        raise NotImplementedError()

    def log_volume(self) -> float:
        raise NotImplementedError()

    def increase(self):
        raise NotImplementedError()

    def __repr__(self):
        trace = np.sum(self.get_tentative_preconditioner())
        logdet = np.sum(np.log(self.get_tentative_preconditioner()))
        return f"Current P: Trace {trace:.2e}, Logdet {logdet:.2e}"


class BoxPreconditionerSet(PreconditionerSet):
    def __init__(self, dim: int, scale: float, incr=None):
        self.b = np.ones(dim) * scale * dim
        INCR_CONST = np.exp(np.log(dim) / dim)
        self.incr = incr if incr is not None else INCR_CONST

    def from_max(self, b: NDArray) -> PreconditionerSet:
        newbox = BoxPreconditionerSet(dim=len(b), scale=1, incr=self.incr)
        newbox.b = b
        return newbox

    def increase(self) -> PreconditionerSet:
        return self.from_max(self.incr * self.b)

    def get_tentative_preconditioner(self, grad: NDArray = None):
        return self.b / len(self.b)

    def cut(self, u: NDArray, refine_steps: bool = 0, P=None):
        # new_b = np.minimum(self.b, 1 / u)
        new_b = 1 / np.maximum(1 / self.b, u)
        return self.from_max(new_b)

    def log_volume(self) -> float:
        return np.sum(np.log(self.b))


class SimplexPreconditionerSet(PreconditionerSet):
    def __init__(self, dim: int, scale: float, incr=None):
        self.b = np.ones(dim) * scale
        INCR_CONST = np.exp(np.log(2 / np.sqrt(np.e)) / dim)
        self.incr = incr if incr is not None else INCR_CONST

    def from_max(self, b: NDArray) -> PreconditionerSet:
        newbox = SimplexPreconditionerSet(dim=len(b), scale=1, incr=self.incr)
        newbox.b = b
        return newbox

    def increase(self) -> PreconditionerSet:
        return self.from_max(self.incr * self.b)

    def get_tentative_preconditioner(self, grad: NDArray = None):
        return self.b / 2

    def cut(self, u: NDArray, refine_steps=0, P=None):
        c = 0.5
        d = self.b.shape[0]
        _lambda = (d - 1) / (d - c)
        v = (1 / self.b) / d
        w = _lambda * v + (1 - _lambda) * u

        if refine_steps > 0:
            a, b = _lambda, 1 - _lambda
            for _ in range(refine_steps):
                a *= np.inner(v, 1 / w)
                b *= np.inner(u, 1 / w)
                a, b = a / (a + b), b / (a + b)
                w = a * v + b * u

        new_b = (1 / w) / d

        return self.from_max(new_b)

    def log_volume(self) -> float:
        return np.sum(np.log(self.b))


class EllipsoidPreconditionerSet(PreconditionerSet):
    def __init__(self, dim: int, scale: float, incr=None):
        self.inverse_diagonal = (np.ones(dim) * (scale * dim) ** 2).astype(float)
        INCR_CONST = np.exp(np.log(np.sqrt(2 / np.sqrt(np.e))) / dim)
        self.incr = incr if incr is not None else INCR_CONST

    def from_inv_diag(self, inv_diag: NDArray) -> PreconditionerSet:
        newbox = EllipsoidPreconditionerSet(dim=len(inv_diag), scale=1, incr=self.incr)
        newbox.inverse_diagonal = inv_diag
        return newbox

    def increase(self) -> PreconditionerSet:
        return self.from_inv_diag(np.sqrt(self.incr) * self.inverse_diagonal)

    def get_tentative_preconditioner(self, grad: NDArray = None):
        d = len(self.inverse_diagonal)
        if grad is None:
            starting_preconditioner = np.sqrt(self.inverse_diagonal)
            P = starting_preconditioner / (np.sqrt(2) * d)
        else:
            norm = np.linalg.norm(np.sqrt(self.inverse_diagonal) * grad**2)
            starting_preconditioner = self.inverse_diagonal * grad**2 / norm
            P = starting_preconditioner / (np.sqrt(2) * np.sqrt(d))
        return P

    def cut(self, u: NDArray, refine_steps=0, P=None):
        _d = self.inverse_diagonal.shape[0]
        _v = 1 / self.inverse_diagonal
        _u2 = u**2
        ell = np.inner(_u2, self.inverse_diagonal)
        _lambda = (ell / _d) * (_d - 1) / (ell - 1)
        # norm-agnostic constant
        # _c = 0.5
        # _lambda = (_d - 1) / (_d - _c)

        if _lambda < 0 or _lambda > 1 or ell < _d:
            import pdb

            pdb.set_trace()

        _w = None
        if refine_steps != 0:
            if refine_steps == "bfgs":
                diff = _v - _u2

                def f_(c):
                    return -np.sum(np.log(c * diff + _u2))

                res = sp.optimize.minimize(
                    fun=f_,
                    x0=_lambda,
                    method="L-BFGS-B",
                    bounds=sp.optimize.Bounds(lb=0.01, ub=_lambda),
                )

                _new_lambda = res.x[0]
                _w = _new_lambda * _v + (1 - _new_lambda) * _u2
            elif refine_steps > 0:
                _w = _lambda * _v + (1 - _lambda) * _u2
                a, b = _lambda, 1 - _lambda
                for _ in range(refine_steps):
                    a *= np.inner(_v, 1 / _w)
                    b *= np.inner(_u2, 1 / _w)
                    a, b = a / (a + b), b / (a + b)
                    _w = a * _v + b * _u2
            else:
                raise ValueError("what")
        else:
            _w = _lambda * _v + (1 - _lambda) * _u2

        with np.errstate(all="raise"):
            new_diag = 1 / _w

        return self.from_inv_diag(new_diag)

    def log_volume(self) -> float:
        return np.sum(np.log(np.sqrt(self.inverse_diagonal)))


##
# Template for preconditioner search


def progress_gap(new_f: float, f: float, P: NDArray, g: NDArray) -> float:
    return new_f - f + 0.5 * np.inner(g, P * g)


def sufficient_progress(new_f: float, f: float, P: NDArray, g: NDArray) -> bool:
    return progress_gap(new_f, f, P, g) <= 0


def sep_hyperplane(Sy: OptProblemAtPoint, Sx: OptProblemAtPoint, P: NDArray) -> NDArray:
    progress_grad = 0.5 * Sx.g**2 - Sy.g * Sx.g
    # unstable
    # normalization = np.inner(progress_grad, P) - progress_gap(Sy.f, Sx.f, P, Sx.g)
    normalization = Sx.f - Sy.f - np.inner(Sy.g, P * Sx.g)
    normalized_hyperplane = progress_grad / normalization

    return normalized_hyperplane


def strong_hyperplane(
    Snew: OptProblemAtPoint, Sold: OptProblemAtPoint, P: NDArray
) -> NDArray:
    return np.maximum(sep_hyperplane(Snew, Sold, P), 0)


BOX = "box"
SIMPLEX = "simplex"
ELLIPSOID = "ellipsoid"


def solve_precsearch(
    func: Callable[[NDArray], float],
    grad: Callable[[NDArray], NDArray],
    x0: NDArray,
    set_type: Literal["box", "simplex", "ellipsoid"],
    initial_box: float = 10**3,
    backtrack: float = 0.5,
    forward: Optional[float] = None,
    grad_dir: bool = False,
    refine: int = 0,
    maxiter: int = 1000,
    tol: float = 10**-3,
    callback: Optional[
        Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
    ] = None,
):
    dim = len(x0)
    if backtrack is not None:
        warnings.warn(
            "NotImplementedWarning: The backtrack parameter is not used. "
            "Current implementation usese hardcoded valud in get_tentative_preconditioner;"
            "1/2d for the box, 1/2 for the simplex and 1/sqrt(2d) for the ellipsoid"
        )
    if set_type == BOX:
        preconditioner_set = BoxPreconditionerSet(
            dim=dim, scale=initial_box, incr=forward
        )
    elif set_type == SIMPLEX:
        preconditioner_set = SimplexPreconditionerSet(
            dim=dim, scale=initial_box, incr=forward
        )
    elif set_type == ELLIPSOID:
        preconditioner_set = EllipsoidPreconditionerSet(
            dim=dim, scale=initial_box, incr=forward
        )
    else:
        raise ValueError(
            f"Unknown set type {set_type}. "
            f"Expected one of [box, simplex, ellipsoid]"
        )

    def iteration(Sx: OptProblemAtPoint, prec_set: PreconditionerSet):
        if grad_dir:
            P = prec_set.get_tentative_preconditioner(Sx.g)
        else:
            P = prec_set.get_tentative_preconditioner()
        Sy = Sx.at_y(Sx.x - P * Sx.g)

        if sufficient_progress(Sy.f, Sx.f, P, Sx.g):
            return Sy, prec_set.increase()
        else:
            u = strong_hyperplane(Sy, Sx, P)

            if np.inner(u, P) < 1:
                expected_progress = np.inner(Sx.g**2, P)
                if expected_progress < 1e-12:
                    raise FloatingPointError(
                        f"Expected progress is too small {expected_progress:.2e},"
                        "causing numerical stability issues."
                    )

            new_set = prec_set.cut(u, refine_steps=refine, P=P)
            if new_set.log_volume() >= prec_set.log_volume():
                raise FloatingPointError(
                    f"Volume did not decrease after a cut, "
                    f"most likely due to numerical issues "
                )

            return Sx, new_set

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
                "prec_set": preconditioner_set,
                "nfev": nfev,
                "ngev": ngev,
                "nbacktrack": nbacktrack,
            },
        )

    for t in range(maxiter):
        try:
            new_prob_at_x, new_preconditioner_set = iteration(
                prob_at_x, preconditioner_set
            )
        except FloatingPointError:
            break

        iteration_type = "backtrack" if new_prob_at_x is prob_at_x else "update"

        if iteration_type == "backtrack":
            nfev += 1
            ngev += 1
            nbacktrack += 1
        elif iteration_type == "update":
            nfev += 1
            ngev += 1

        prob_at_x, preconditioner_set = new_prob_at_x, new_preconditioner_set

        if np.linalg.norm(prob_at_x.g) <= tol:
            break

        if callback is not None:
            callback(
                prob_at_x,
                {
                    "prec_set": preconditioner_set,
                    "nfev": nfev,
                    "ngev": ngev,
                    "nbacktrack": nbacktrack,
                },
            )

    return prob_at_x
