"""
Wrappers for the optimization classes for integration in experiment library
"""
import dataclasses
from typing import Any, Callable, Dict, Literal, Optional

import numpy as np
import scipy as sp
import scipy.sparse.linalg

from precsearch import optimal_preconditioner
from precsearch.optimizers.base_optimizers import (
    solve_diag_h,
    solve_lbfgs,
    solve_preconditioned_gd_ls,
)
from precsearch.optimizers.diagonalbb import solve_diagonalbb
from precsearch.optimizers.hypergd import solve_hypergd
from precsearch.optimizers.online_learning_optimizers import (
    solve_adagrad,
    solve_adagrad_ls,
)
from precsearch.optimizers.preconditioner_search import solve_precsearch
from precsearch.optimizers.rmsprop import solve_rmsprop
from precsearch.optimizers.rprop import solve_rprop
from precsearch.problems import LinearRegression, OptProblemAtPoint, Problem


@dataclasses.dataclass
class Optimizer:
    def __init__(self):
        pass

    def uname(self):
        raise NotImplementedError

    def get_solve_func(
        self, problem: Problem
    ) -> Callable[
        [
            OptProblemAtPoint,
            Optional[Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]],
        ],
        OptProblemAtPoint,
    ]:
        raise NotImplementedError


@dataclasses.dataclass
class GDLS(Optimizer):
    maxiter: int = 100
    tol: float = 0
    init_ss: float = 1.0
    backtrack: float = 0.5
    forward: float = 2.0

    def uname(self):
        return (
            f"GDLS("
            f"init={self.init_ss},"
            f"backtrack={self.backtrack},"
            f"forward={self.forward},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_preconditioned_gd_ls(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                tol=self.tol,
                starting_stepsize=self.init_ss,
                backtrack=self.backtrack,
                forward=self.forward,
                preconditioner=None,
                maxiter=self.maxiter,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class DiagH(Optimizer):
    maxiter: int = 100
    tol: float = 0
    init_ss: float = 1.0
    backward: float = 0.5
    forward: float = 1.1

    def uname(self):
        return (
            f"DiagH("
            f"init={self.init_ss},"
            f"forward={self.forward},"
            f"backward={self.backward},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_diag_h(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                diag_h=prob_at_x.problem.diag_hess,
                x0=prob_at_x.x,
                tol=self.tol,
                forward=self.forward,
                backward=self.backward,
                stepsize=self.init_ss,
                maxiter=self.maxiter,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class OptPGDLS(Optimizer):
    maxiter: int = 100
    tol: float = 0
    init_ss: float = 1.0
    backtrack: float = 0.5
    forward: float = 2.0

    def uname(self):
        return (
            f"OptPGDLS("
            f"init={self.init_ss},"
            f"backtrack={self.backtrack},"
            f"forward={self.forward},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            X, y = problem.dataset.load()

            bias_col = np.ones((X.shape[0], 1))

            if sp.sparse.issparse(X):
                X = sp.sparse.hstack((X, bias_col))
            else:
                X = np.hstack((X, bias_col))

            n, d = X.shape
            reg = problem.regularization

            hessian = (X.T @ X + reg * np.eye(d)) / n

            P = optimal_preconditioner(hessian)
            preconditioner = sp.sparse.linalg.aslinearoperator(np.diag(P**2))

            return solve_preconditioned_gd_ls(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                tol=self.tol,
                starting_stepsize=self.init_ss,
                backtrack=self.backtrack,
                forward=self.forward,
                preconditioner=preconditioner,
                maxiter=self.maxiter,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class OptPGD(Optimizer):
    maxiter: int = 100
    tol: float = 0

    def uname(self):
        return f"OptPGD(" f"maxiter={self.maxiter}," f"tol={self.tol})"

    def get_solve_func(self, problem: Problem):
        return OptPGDLS(
            maxiter=self.maxiter,
            tol=self.tol,
            init_ss=1.0,
            backtrack=1.0,
            forward=1.0,
        ).get_solve_func(problem)


@dataclasses.dataclass
class GD(Optimizer):
    maxiter: int = 100
    tol: float = 0
    stepsize: float = 1.0

    def uname(self):
        return (
            f"GD("
            f"stepsize={self.stepsize},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        return GDLS(
            maxiter=self.maxiter,
            tol=self.tol,
            init_ss=self.stepsize,
            backtrack=1.0,
            forward=1.0,
        ).get_solve_func(problem)


@dataclasses.dataclass
class OptGD(Optimizer):
    maxiter: int = 100
    tol: float = 0

    def uname(self):
        return f"GD(" f"maxiter={self.maxiter}," f"tol={self.tol})"

    def get_solve_func(self, problem: Problem):
        if not isinstance(problem, LinearRegression):
            raise ValueError(
                "Optimal Preconditioning is not implemented for this problem. "
                f"Got {problem.__class__}, Expected {LinearRegression}"
            )

        X, y = problem.dataset.load()
        n, d = X.shape
        reg = problem.regularization

        hessian = (X.T @ X + reg * np.eye(d)) / n
        stepsize = 1 / np.max(np.linalg.eigvalsh(hessian))

        return GD(maxiter=self.maxiter, tol=self.tol, stepsize=stepsize).get_solve_func(
            problem
        )


@dataclasses.dataclass
class PrecSearch(Optimizer):
    """
    Preconditioner search

    Properties:
        set_type: The set type to be used for the preconditioner search
          (either "box", "simplex", or "ellipsoid").
        maxiter: The maximum number of iterations
        tol: A float indicating the tolerance level for convergence.
        initial_box: A float indicating the initial size of the set for the preconditioner search.
        backtrack: How much to backtrack
        forward: How much to increase the size of the set by.
        grad_dir: Whether to use the gradient direction to select the preconditioner
          (only used for the ellipsoid method)
        refine_steps: The number of refinement for the minimum volume set.
          (only used for the simplex and ellipsoid method)
    """

    set_type: Literal["box", "simplex", "ellipsoid"]
    maxiter: int = 100
    tol: float = 0
    initial_box: float = 1.0
    backtrack: float = 0.5
    forward: float = 2.0
    grad_dir: bool = True
    refine_steps: int = 0

    def uname(self):
        return (
            f"PrecSearch[{self.set_type}]("
            f"init={self.initial_box},"
            f"backtrack={self.backtrack},"
            f"forward={self.forward},"
            f"grad_dir={self.grad_dir})"
            f"refine={self.refine_steps})"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_precsearch(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                set_type=self.set_type,
                initial_box=self.initial_box,
                backtrack=self.backtrack,
                forward=self.forward,
                grad_dir=self.grad_dir,
                refine=self.refine_steps,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class LBFGS(Optimizer):
    maxiter: int = 100
    tol: float = 0
    L: int = 10
    verbose: bool = False

    def uname(self):
        return f"LBFGS(" f"L={self.L}" f"maxiter={self.maxiter}," f"tol={self.tol})"

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_lbfgs(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                L=self.L,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
                iprint=1 if self.verbose else 0,
            )

        return solve


@dataclasses.dataclass
class RPROP(Optimizer):
    """
    RPROP


    Properties:
        starting_stepsize: A float initial step-size to use on each coordinate
        maxiter: The maximum number of iterations
        tol: A float indicating the tolerance level for convergence.
        eta_plus:
        eta_minus:
        max_stepsize:
        min_stepsize:

    """

    starting_stepsize: float = 10**-2
    maxiter: int = 100
    eta_plus: float = 1.2
    eta_minus: float = 0.5
    max_stepsize: float = 50
    min_stepsize: float = 10**-6
    tol: float = 10**-3

    def uname(self):
        return (
            f"RPROP("
            f"eta_plus={self.eta_plus},"
            f"eta_minus={self.eta_minus},"
            f"max_stepsize={self.max_stepsize},"
            f"min_stepsize={self.min_stepsize}"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_rprop(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                starting_stepsize=self.starting_stepsize,
                eta_plus=self.eta_plus,
                eta_minus=self.eta_minus,
                max_stepsize=self.max_stepsize,
                min_stepsize=self.min_stepsize,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class AdaGradNorm(Optimizer):
    starting_stepsize: float = 10**-2
    maxiter: int = 100
    tol: float = 10**-3

    def uname(self):
        return (
            f"AdaGradNorm("
            f"starting_stepsize={self.starting_stepsize},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_adagrad(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                starting_stepsize=self.starting_stepsize,
                diagonal=False,
                D=None,
                project=False,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class AdaGrad(Optimizer):
    starting_stepsize: float = 10**-2
    maxiter: int = 100
    tol: float = 10**-3

    def uname(self):
        return (
            f"AdaGrad("
            f"starting_stepsize={self.starting_stepsize},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_adagrad(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                starting_stepsize=self.starting_stepsize,
                diagonal=True,
                D=None,
                project=False,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class AdaGradLS(Optimizer):
    maxiter: int = 100
    tol: float = 10**-3

    def uname(self):
        return f"AdaGradLS(" f"maxiter={self.maxiter}," f"tol={self.tol})"

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_adagrad_ls(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class HyperGD(Optimizer):
    """
    HyperGD

    Properties:
        starting_stepsize: A float initial step-size to use
        hyper_stepsize: A flot step-size to use on gradient descent on the
         step-size itself
        maxiter: The maximum number of iterations
        backtrack: Backtracking factor for setting initial step-size
        tol: A float indicating the tolerance level for convergence.
        multiplicative_update: A bool indicating whether to use multiplicative
         updates in the hypergreadient descent steps. Default value is False.
         Note that the original paper suggests setting starting_steps to 0.02
         when using multiplicative updates


    """

    maxiter: int = 100
    starting_stepsize: float = 10**-2
    hyper_stepsize: float = 10**-4
    backtrack: float = 0.5
    tol: float = 10**-3
    multiplicative_update: bool = False

    def uname(self):
        return (
            f"HyperGD("
            f"starting_stepsize={self.starting_stepsize},"
            f"hyper_stepsize={self.hyper_stepsize},"
            f"backtrack={self.backtrack},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_hypergd(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                starting_stepsize=self.starting_stepsize,
                hyper_stepsize=self.hyper_stepsize,
                backtrack=self.backtrack,
                maxiter=self.maxiter,
                tol=self.tol,
                multiplicative_update=self.multiplicative_update,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class HyperGDAdd(Optimizer):
    maxiter: int = 100
    tol: float = 10**-3
    backtrack: float = 0.5

    def uname(self):
        return f"HyperGDAdd(maxiter={self.maxiter}," f"tol={self.tol})"

    def get_solve_func(self, problem: Problem):
        return HyperGD(
            starting_stepsize=10**-10,
            hyper_stepsize=10**-4,
            multiplicative_update=False,
            backtrack=self.backtrack,
            tol=self.tol,
            maxiter=self.maxiter,
        ).get_solve_func(problem)


@dataclasses.dataclass
class HyperGDMult(Optimizer):
    maxiter: int = 100
    tol: float = 10**-3
    backtrack: float = 0.5

    def uname(self):
        return f"HyperGDMult(maxiter={self.maxiter}," f"tol={self.tol})"

    def get_solve_func(self, problem: Problem):
        return HyperGD(
            starting_stepsize=10**-10,
            hyper_stepsize=2e-2,
            multiplicative_update=True,
            backtrack=self.backtrack,
            tol=self.tol,
            maxiter=self.maxiter,
        ).get_solve_func(problem)


@dataclasses.dataclass
class DiagonalBB(Optimizer):
    """
    DiagonalBB

    Source: "VARIABLE METRIC PROXIMAL GRADIENT METHOD WITH DIAGONAL
        BARZILAI-BORWEIN STEPSIZE" by Park, Dhar, Boyd, and Shah from 2020

    Properties:
        starting_stepsize: A float initial step-size to use (with U = identity at first)
        mu: Regularization parameter to prevent the preconditioner U from changing too abruptly
        backtrack: Backtracking parameter for the linesearch
        ls_window: length of the window of values used on the non-monotonic line-search
        maxiter: The maximum number of iterations
        tol: A float indicating the tolerance level for convergence.
    """

    maxiter: int = 100
    starting_stepsize: float = 10**-6
    mu: float = 10**-6
    backtrack: float = 0.5
    ls_window: int = 15
    tol: float = 10**-3

    def uname(self):
        return (
            f"DiagonalBB("
            f"starting_stepsize={self.starting_stepsize},"
            f"mu={self.mu},"
            f"backtrack={self.backtrack},"
            f"maxiter={self.maxiter},"
            f"ls_window={self.ls_window},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_diagonalbb(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                starting_stepsize=self.starting_stepsize,
                mu=self.mu,
                backtrack=self.backtrack,
                maxiter=self.maxiter,
                ls_window=self.ls_window,
                tol=self.tol,
                callback=callback,
            )

        return solve


@dataclasses.dataclass
class RMSProp(Optimizer):
    """
    RMSProp

    Properties:
        starting_stepsize: A float initial step-size to use
        avg_decay: Float decay factor for the moving average of squared grandients
        denominator_offset: Positive float summed do avg_decay when using it to normalize the gradient step
        backtrack: Backtracking parameter for the starting linesearch
        maxiter: The maximum number of iterations
        tol: A float indicating the tolerance level for convergence.
    """

    maxiter: int = 100
    avg_decay: float = 0.9
    denominator_offset: float = 10**-8
    tol: float = 10**-3

    def uname(self):
        return (
            f"RMSProp("
            f"avg_decay={self.avg_decay},"
            f"denominator_offset={self.denominator_offset},"
            f"maxiter={self.maxiter},"
            f"tol={self.tol})"
        )

    def get_solve_func(self, problem: Problem):
        def solve(
            prob_at_x: OptProblemAtPoint,
            callback: Optional[
                Callable[[OptProblemAtPoint, Dict[str, Any]], Optional[bool]]
            ] = OptProblemAtPoint,
        ):
            return solve_rmsprop(
                func=prob_at_x.problem.func,
                grad=prob_at_x.problem.grad,
                x0=prob_at_x.x,
                avg_decay=self.avg_decay,
                denominator_offset=self.denominator_offset,
                maxiter=self.maxiter,
                tol=self.tol,
                callback=callback,
            )

        return solve
