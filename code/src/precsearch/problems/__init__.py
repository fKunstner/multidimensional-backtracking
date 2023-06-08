import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import scipy as sp
import scipy.sparse
from numpy.typing import NDArray

from precsearch.datasets import Dataset
from precsearch.initializations import Initializer


@dataclasses.dataclass
class OptProblem:
    """
    A class representing an optimization problem.

    Attributes:
        func: A callable that takes an array of shape (d,) as input
            and returns a scalar value.
        grad: A callable that takes an array of shape (d,) as input
            and returns an array of shape (d,) representing the
            gradient of the function at that point.
        n: An optional integer representing the number of examples
            in the dataset.
        d: An optional integer representing the dimensionality of
            the feature space.
    """

    func: Callable[[NDArray], float]
    grad: Callable[[NDArray], NDArray]
    diag_hess: Optional[Callable[[NDArray], NDArray]] = None
    n: int = None
    d: int = None


@dataclasses.dataclass
class OptProblemAtPoint:
    """
    A class representing an optimization problem at a specific point.

    Attributes:
        problem: An instance of the OptProblem class representing the
            optimization problem.
        x: An array of shape (d,) representing the point at which to
            evaluate the problem.

    Properties:
        f: A cached property representing the value of the objective
            function at the point `x`.
        g: A cached property representing the gradient of the objective
            function at the point `x`.

    Methods:
        at_y(y: NDArray) -> OptProblemAtPoint:
            Returns a new OptProblemAtPoint instance with the same
            problem but with the point `x` replaced by the input `y`.
    """

    problem: OptProblem
    x: NDArray

    @functools.cached_property
    def f(self):
        return self.problem.func(self.x)

    @functools.cached_property
    def g(self):
        return self.problem.grad(self.x)

    @functools.cached_property
    def diag_h(self):
        return self.problem.diag_hess(self.x)

    def at_y(self, y: NDArray):
        return OptProblemAtPoint(self.problem, y)


@dataclass
class Problem:
    """
    Wraps a dataset, model and objective function to define an optimization problem.

    Functions are implemented as the average over data points
    Regularization is also rescaled by the number of samples
    (strength is expressed in pseudo-observations)
    """

    dataset: Dataset
    init: Initializer
    regularization: float = 0
    model_type: str = field(init=False)

    def opt_problem(self) -> OptProblemAtPoint:
        X, y = self.dataset.load()
        n, d = X.shape

        bias_col = np.ones((n, 1))

        if sp.sparse.issparse(X):
            X = sp.sparse.hstack((X, bias_col))
        else:
            X = np.hstack((X, bias_col))

        d = d + 1

        x0 = self.init.initialize(X, y)

        return OptProblemAtPoint(
            problem=OptProblem(
                func=self.make_f(X, y),
                grad=self.make_g(X, y),
                diag_hess=self.make_diag_h(X, y),
                n=n,
                d=d,
            ),
            x=x0,
        )

    def make_f(self, X, y):
        raise NotImplementedError

    def make_g(self, X, y):
        raise NotImplementedError

    def make_diag_h(self, X, y):
        raise NotImplementedError

    def make_regularizer(self):
        def reg_f(w):
            return self.regularization * 0.5 * np.linalg.norm(w) ** 2

        return reg_f

    def make_regularizer_grad(self):
        def reg_g(w):
            return self.regularization * w

        return reg_g

    def make_regularizer_diag_h(self):
        def reg_diag_h(w):
            return self.regularization * np.ones_like(w)

        return reg_diag_h

    def uname(self):
        try:
            return f"{self.model_type}(dataset={self.dataset.uname()},init={self.init.uname()},reg={self.regularization})"
        except Exception as e:
            import pdb

            pdb.set_trace()


@dataclass
class LinearRegression(Problem):
    def __post_init__(self):
        self.model_type = "LinReg"

    def make_f(self, X, y):
        reg = self.make_regularizer()

        def f(w):
            loss = 0.5 * np.linalg.norm((X @ w - y)) ** 2
            r = reg(w)
            return (loss + r) / X.shape[0]

        return f

    def make_g(self, X, y):
        reg_g = self.make_regularizer_grad()

        def g(w):
            loss_grad = X.T @ (X @ w - y)
            r_grad = reg_g(w)
            return (loss_grad + r_grad) / X.shape[0]

        return g

    def make_diag_h(self, X, y):
        reg_diag_h = self.make_regularizer_diag_h()

        def diag_h(w):
            if sp.sparse.issparse(X):
                Z = X.multiply(X)
            else:
                Z = np.multiply(X, X)
            loss_diag_h = np.sum(Z, axis=0)

            r_diag_h = reg_diag_h(w)
            DH = (loss_diag_h + r_diag_h) / X.shape[0]
            return np.asarray(DH).reshape((-1))

        return diag_h


def logsig(x):
    """Compute the log-sigmoid function component-wise.

    https://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    idx1 = (x >= -33) & (x < -18)
    idx2 = (x >= -18) & (x < 37)
    idx3 = x >= 37
    out[idx0] = x[idx0]
    out[idx1] = x[idx1] - np.exp(x[idx1])
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    out[idx3] = -np.exp(-x[idx3])
    return out


def expit_b(x, b):
    """Compute sigmoid(x) - b component-wise.

    https://fa.bianp.net/blog/2019/evaluate_logistic/
    """
    out = np.zeros_like(x)
    idx_neg = x < 0
    exp_x = np.exp(x[idx_neg])
    exp_nx = np.exp(-x[~idx_neg])
    out[idx_neg] = ((1 - b[idx_neg]) * exp_x - b[idx_neg]) / (1 + exp_x)
    out[~idx_neg] = ((1 - b[~idx_neg]) - b[~idx_neg] * exp_nx) / (1 + exp_nx)
    return out


@dataclass
class LogisticRegression(Problem):
    def __post_init__(self):
        self.model_type = "LogReg"

    def make_f(self, X, y):
        reg = self.make_regularizer()

        def f(w):
            z = X @ w
            return np.mean((1 - y) * z - logsig(z)) + reg(w) / X.shape[0]

        return f

    def make_g(self, X, y):
        reg_g = self.make_regularizer_grad()

        def g(w):
            z = X.dot(w)
            s = expit_b(z, y)
            return X.T.dot(s) / X.shape[0] + reg_g(w) / X.shape[0]

        return g

    def make_diag_h(self, X, y):
        reg_diag_h = self.make_regularizer_diag_h()

        def diag_h(w):
            z = X.dot(w)
            probs = expit_b(z, np.zeros_like(y))
            weights = np.multiply(probs, 1 - probs)

            if sp.sparse.issparse(X):
                Z = X.multiply(X)
                loss_diag = weights * Z
            else:
                Z = np.multiply(X, X)
                loss_diag = (weights * Z).reshape((-1,))

            DH = (loss_diag + reg_diag_h(w)) / X.shape[0]
            return np.asarray(DH).reshape((-1))

        return diag_h
