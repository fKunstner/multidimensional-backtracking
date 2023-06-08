import importlib
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

import precsearch
import precsearch.constants
import precsearch.optimizers
import precsearch.plotting
import precsearch.plotting_style
from precsearch.datasets import Dataset
from precsearch.initializations import GaussianInitializer
from precsearch.problems import LinearRegression, Problem


def load_data():
    importlib.reload(precsearch)

    problems: List[Problem] = [
        LinearRegression(
            dataset=Dataset(dataset),
            regularization=1.0,
            init=GaussianInitializer(),
        )
        for dataset in [
            "breast-cancer",
            "heart",
            "australian",
            "diabetes",
            "ionosphere",
            "california-housing",
            "concrete",
            "power-plant",
            "cpusmall",
            "mg",
        ]
    ]

    return {"problems": problems}


def postprocess(data):
    for prob in data["problems"]:
        if isinstance(prob, Problem):
            X, y = prob.dataset.load()
            prob.opt_problem()

            n, d = np.shape(X)
            bias_col = np.ones((n, 1))

            if sp.sparse.issparse(X):
                X = sp.sparse.hstack((X, bias_col))
            else:
                X = np.hstack((X, bias_col))

            d = d + 1

            reg = prob.regularization

            if d < 50:
                hessian = (X.T @ X + reg * np.eye(d)) / n
                sqrtDH = np.diag(1 / np.sqrt(np.diag(hessian)))
                P = precsearch.optimal_preconditioner(hessian)
                sqrtP = np.diag((P))
                print(prob.dataset.dataset_name)
                print(
                    "    Original         condition number:",
                    f"{np.linalg.cond(hessian):.2e}",
                )
                print(
                    "     Diagonal-Hessian condition number:",
                    f"{np.linalg.cond(sqrtDH @ hessian @ sqrtDH):.2e}",
                )
                print(
                    "    Optimal          condition number:",
                    f"{np.linalg.cond(sqrtP @ hessian @ sqrtP):.2e}",
                )
            else:
                if n > d:
                    raise NotImplementedError

                def maxEig(matvec):
                    sp.sparse.linalg.LinearOperator(shape=(d, d), matvec=matvec)
                    return sp.sparse.linalg.eigsh(
                        matvec, k=1, return_eigenvectors=False
                    )[0]

                max_eig_h = maxEig(lambda v: (X.T @ (X @ v) + reg * v) / n)
                print(
                    prob.dataset.dataset_name,
                    "         " "Original         condition number:",
                    f"{n*max_eig_h:.2e}",
                )


def settings(plt):
    pass


def make_figure(fig, data):
    pass


if __name__ == "__main__":
    settings(plt)

    fig = plt.figure()
    make_figure(fig, postprocess(load_data()))
    plt.close()
