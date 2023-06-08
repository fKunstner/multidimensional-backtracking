import warnings

import cvxpy as cp
import numpy as np
from cvxpy import SolverError


def optimal_preconditioner(M, verbose=False):
    """
    Given a matrix M, return a diagonal preconditioner P
    that minimizes the condition number of P @ M @ P
    and satisfies P @ M @ P <<= I

    CVXPY implementation of the SDP formulation of Qu et al.,
    > Optimal Diagonal Preconditioning
    > Zhaonan Qu, Wenzhi Gao, Oliver Hinder, Yinyu Ye, Zhengyuan Zhou
    > https://arxiv.org/abs/2209.00809

    Adapted from the CVX implementation in matlab
    https://github.com/Gwzwpxz/opt_dpcond/blob/main/utils/getcvxdiag.m
    """
    D = cp.Variable((M.shape[0], 1), nonneg=True)
    tau = cp.Variable(nonneg=True)
    problem = cp.Problem(
        cp.Maximize(tau),
        constraints=[
            cp.diag(D) << M,
            M * tau << cp.diag(D),
            tau <= 1.0,
        ],
    )
    #   try:
    problem.solve(
        solver="CVXOPT",
        verbose=verbose,
        max_iters=10000,
        abstol=1e-6,
        reltol=1e-6,
        feastol=1e-6,
    )
    #    except SolverError as e:
    #        import pdb
    #
    #        pdb.set_trace()
    #        print(e)

    P = 1 / np.sqrt(D.value.reshape((-1,)))

    lambda_max = np.max(np.linalg.eigvalsh(np.diag(P) @ M @ np.diag(P)))
    P = P / np.sqrt(lambda_max)

    return P


def perturb(M, desired_cond=10**10, eps=10**-6, maxiter=100):
    """Reduces the condition number of M by adding a multiple of the identity."""
    newM = M.copy()
    c = eps
    I = np.eye(newM.shape[0])
    t = 0
    while np.linalg.cond(newM) > desired_cond:
        newM = M + c * I
        c *= 2
        t += 1
        if t > maxiter:
            warnings.warn(
                "Max iterations reached. "
                "Matrix M might not have the desired condition number. "
                "Increase epsilon of the number of iterations."
            )
            break
    return newM
