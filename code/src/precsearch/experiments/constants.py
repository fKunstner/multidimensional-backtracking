import json
import os

import numpy as np

from precsearch import config
from precsearch.experiments.rate_limited_logger import RateLimitedLogger
from precsearch.optimizers.optimizers import LBFGS
from precsearch.problems import Problem


def load_solution(problem: Problem):
    with open(config.problem_info_filepath(problem), "r") as f:
        try:
            data = json.load(f)
        except Exception as e:
            import pdb

            pdb.set_trace()
    return data


def solve_to_high_accuracy(problem: Problem, linear_solve=True, tol=1e20):
    lbfgs = LBFGS(maxiter=10**4, tol=tol, L=10, verbose=True)
    solve = lbfgs.get_solve_func(problem)

    t = 0
    logger = RateLimitedLogger(time_interval=1)

    def callback(prob, state):
        nonlocal t
        t = t + 1
        logger.log({"t": t, "f": prob.f})

    end = solve(problem.opt_problem(), callback)

    if linear_solve:
        X, y = problem.dataset.load()
        n, d = X.shape
        hessian = (X.T @ X) + problem.regularization * np.eye(d)
        linear_system_solution = np.linalg.solve(hessian, X.T @ y)
        end_linear_system_solution = end.at_y(linear_system_solution)

        if end_linear_system_solution.f < end.f:
            end = end_linear_system_solution

    with open(config.problem_info_filepath(problem), "w") as f:
        json.dump({"f": end.f}, f)


def solve_new_for_high_acc(problems, linear_solve=False, tol=0):
    for problem in problems:
        exp_filepath = config.problem_info_filepath(problem)
        if not os.path.isfile(exp_filepath):
            solve_to_high_accuracy(problem, linear_solve=linear_solve, tol=tol)
