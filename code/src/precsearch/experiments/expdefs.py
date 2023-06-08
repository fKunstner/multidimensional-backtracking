from typing import List

from precsearch.datasets import Dataset
from precsearch.experiments import Experiment
from precsearch.initializations import (
    BiasInitializerLinReg,
    BiasInitializerLogReg,
    GaussianInitializer,
)
from precsearch.optimizers.optimizers import (
    GDLS,
    LBFGS,
    RPROP,
    AdaGrad,
    AdaGradLS,
    AdaGradNorm,
    DiagH,
    DiagonalBB,
    HyperGDAdd,
    HyperGDMult,
    Optimizer,
    OptPGDLS,
    PrecSearch,
    RMSProp,
)
from precsearch.optimizers.preconditioner_search import BOX, ELLIPSOID, SIMPLEX
from precsearch.problems import LinearRegression, LogisticRegression, Problem

small_linear_regression_datasets = [
    "cpusmall",
    #    "eunite2001",
    "california-housing",
    #    "pyrim",
    #    "mpg",
    #    "cadata",
    "concrete",
    "power-plant",
    "mg",
    "naval-propulsion",
    #    "bodyfat",
    #    "abalone",
    "yacht",
]

small_logistic_regression_datasets = [
    "breast-cancer",
    "ionosphere",
    "australian",
    "diabetes",
    "heart",
]


large_logistic_regression_datasets = [
    "rcv1.binary",
    "news20.binary",
]

MAXITER_SMALL = 1000
MAXITER_MEDIUM = 10000


def gd_for(maxiter=1000) -> List[Optimizer]:
    return [
        GDLS(maxiter=maxiter, tol=0, init_ss=1e10, backtrack=0.5, forward=forward)
        for forward in [1.0, 1.1, 2.0]
    ]


def precsearch_for(maxiter=1000) -> List[Optimizer]:
    box_precsearch = [
        PrecSearch(
            set_type=BOX,
            maxiter=maxiter,
            tol=0,
            initial_box=1e10,
            backtrack=0.5,
            forward=forward,
        )
        for forward in [1.0, 1.1, 2.0]
    ]
    ellipsoid_precsearch = [
        PrecSearch(
            set_type=ELLIPSOID,
            maxiter=maxiter,
            tol=0,
            initial_box=1e10,
            backtrack=0.5,
            forward=forward,
            refine_steps=refine_steps,
        )
        for forward in [1.0, 1.1, 2.0]
        for refine_steps in [0, "bfgs"]
    ]
    return box_precsearch + ellipsoid_precsearch


def heuristics_for(maxiter=1000):
    heuristics = [
        RPROP(maxiter=maxiter, tol=0),
        HyperGDMult(maxiter=maxiter, tol=0),
        HyperGDAdd(maxiter=maxiter, tol=0),
        RMSProp(maxiter=maxiter, tol=0),
    ]
    return heuristics


def OL_opts_for(maxiter=1000):
    opts = [
        AdaGradNorm(starting_stepsize=10**-2, maxiter=maxiter, tol=0),
        AdaGrad(starting_stepsize=10**-2, maxiter=maxiter, tol=0),
        AdaGradLS(maxiter=maxiter, tol=0),
    ]
    return opts


def secondorder_opts_for(maxiter=1000):
    opts = [
        DiagH(init_ss=0.1, maxiter=maxiter, tol=0),
        DiagonalBB(maxiter=maxiter, tol=0),
        LBFGS(maxiter=maxiter, tol=0),
    ]
    return opts


def all_opts_for(maxiter=1000):
    return (
        gd_for(maxiter)
        + precsearch_for(maxiter)
        + heuristics_for(maxiter)
        + OL_opts_for(maxiter)
        + secondorder_opts_for(maxiter)
    )


toy_linear_regression_problems: List[Problem] = [
    LinearRegression(
        dataset=Dataset(dataset_name),
        regularization=1.0,
        init=init,
    )
    for dataset_name in small_linear_regression_datasets
    for init in [GaussianInitializer(var=1.0, seed=0), BiasInitializerLinReg()]
]


small_logistic_regression_problems: List[Problem] = [
    LogisticRegression(
        dataset=Dataset(dataset_name),
        regularization=1.0,
        init=init,
    )
    for dataset_name in small_logistic_regression_datasets
    for init in [BiasInitializerLogReg()]
]

large_logistic_regression_problems: List[Problem] = [
    LogisticRegression(
        dataset=Dataset(dataset_name),
        regularization=1.0,
        init=init,
    )
    for dataset_name in small_linear_regression_datasets
    for init in [BiasInitializerLogReg()]
]

experiments_toy_linear_regression: List[Experiment] = [
    Experiment(
        prob=prob,
        opt=opt,
    )
    for prob in toy_linear_regression_problems
    for opt in all_opts_for(maxiter=1000)
]

experiments_toy_linear_regression_opt_preconditioner: List[Experiment] = [
    Experiment(
        prob=LinearRegression(
            dataset=Dataset(dataset_name),
            regularization=1.0,
            init=init,
        ),
        opt=OptPGDLS(
            maxiter=maxiter,
            tol=0,
            init_ss=1.0,
            backtrack=1.0,
            forward=1.0,
        ),
    )
    for dataset_name in small_linear_regression_datasets
    for init in [GaussianInitializer(var=1.0, seed=0), BiasInitializerLinReg()]
    for maxiter in [MAXITER_SMALL, MAXITER_MEDIUM]
]
