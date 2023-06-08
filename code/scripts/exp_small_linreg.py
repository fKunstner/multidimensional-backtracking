from typing import List

from precsearch.datasets import Dataset
from precsearch.experiments import Experiment, run_all_new_experiments
from precsearch.experiments.constants import solve_new_for_high_acc
from precsearch.experiments.expdefs import all_opts_for
from precsearch.initializations import BiasInitializerLinReg, GaussianInitializer
from precsearch.problems import LinearRegression, Problem

MAXITER = 2000
INIT_SS = 1e10

problems: List[Problem] = [
    LinearRegression(
        dataset=Dataset(dataset),
        regularization=1.0,
        init=init,
    )
    for dataset in [
        "cpusmall",
        "california-housing",
        "concrete",
        "power-plant",
        "mg",
    ]
    for init in [BiasInitializerLinReg(), GaussianInitializer()]
]

experiments = [
    Experiment(prob=prob, opt=opt) for prob in problems for opt in all_opts_for(MAXITER)
]

if __name__ == "__main__":
    solve_new_for_high_acc(problems)
    run_all_new_experiments([exp for exp in experiments])
