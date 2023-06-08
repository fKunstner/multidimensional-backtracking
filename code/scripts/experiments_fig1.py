import os
from typing import List

from precsearch import config
from precsearch.datasets import Dataset
from precsearch.experiments import Experiment, run_all_new_experiments
from precsearch.experiments.constants import solve_to_high_accuracy
from precsearch.experiments.expdefs import small_linear_regression_datasets
from precsearch.initializations import BiasInitializerLinReg
from precsearch.optimizers.optimizers import (
    GDLS,
    OptGD,
    Optimizer,
    OptPGD,
    OptPGDLS,
    PrecSearch,
)
from precsearch.optimizers.preconditioner_search import BOX, ELLIPSOID
from precsearch.problems import LinearRegression, Problem

MAXITER = 10000
INIT_SS = 1e10

optimizers: List[Optimizer] = [
    OptGD(maxiter=MAXITER, tol=0),
    GDLS(maxiter=MAXITER, tol=0, init_ss=INIT_SS, backtrack=0.5, forward=1.0),
    OptPGD(maxiter=MAXITER, tol=0),
    OptPGDLS(maxiter=MAXITER, tol=0, init_ss=INIT_SS, backtrack=0.5, forward=1.0),
    PrecSearch(
        set_type=BOX,
        maxiter=MAXITER,
        tol=0,
        initial_box=INIT_SS,
        backtrack=0.5,
        forward=1.0,
        refine_steps=0,
    ),
    PrecSearch(
        set_type=ELLIPSOID,
        maxiter=MAXITER,
        tol=0,
        initial_box=INIT_SS,
        backtrack=0.5,
        forward=1.0,
        refine_steps=0,
    ),
]

problems: List[Problem] = [
    LinearRegression(
        dataset=Dataset(dataset),
        regularization=1.0,
        init=BiasInitializerLinReg(),
    )
    for dataset in small_linear_regression_datasets
]

experiments = [
    Experiment(prob=prob, opt=opt) for prob in problems for opt in optimizers
]


if __name__ == "__main__":
    for problem in problems:
        exp_filepath = config.problem_info_filepath(problem)
        if not os.path.isfile(exp_filepath):
            solve_to_high_accuracy(problem)

    run_all_new_experiments(experiments)
