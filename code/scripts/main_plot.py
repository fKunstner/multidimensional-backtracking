import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from liveplot import module_loader
from tqdm import tqdm

import precsearch
import precsearch.constants
import precsearch.optimizers
import precsearch.plotting
import precsearch.plotting_style
from precsearch import config
from precsearch.experiments.constants import load_solution
from precsearch.initializations import BiasInitializerLinReg, BiasInitializerLogReg
from precsearch.optimizers.optimizers import (
    GDLS,
    RPROP,
    AdaGradLS,
    DiagH,
    DiagonalBB,
    HyperGDMult,
    OptPGDLS,
    PrecSearch,
)
from precsearch.problems import LinearRegression, LogisticRegression


def load_data():
    importlib.reload(precsearch)
    exps = []
    for file in tqdm(
        [
            "exp_big_logreg.py",
            "exp_small_linreg.py",
            "exp_small_logreg.py",
        ]
    ):
        exps += module_loader._import_module(Path(file)).experiments

    return {"exps": exps}


def postprocess(data):
    data["results"] = {}
    for exp in tqdm(data["exps"]):
        data["results"][exp.uname()] = exp.load()

    unique_problems = {exp.prob.uname(): exp.prob for exp in data["exps"]}

    data["problems_data"] = {
        name: load_solution(problem)
        if config.problem_info_filepath(problem).is_file()
        else print("problem not found")
        for name, problem in unique_problems.items()
    }

    return data


def settings(plt):
    import importlib

    importlib.reload(precsearch.plotting_style)

    plt.rcParams.update(
        precsearch.plotting_style.neurips_config(
            rel_width=1.0, nrows=1, ncols=2, height_to_width_ratio=0.5
        )
    )


def make_figure(fig, data):
    importlib.reload(precsearch.constants)

    fig.set_dpi(250)

    to_plot = [
        {
            "problem": LinearRegression,
            "dataset": "cpusmall",
            "init": BiasInitializerLinReg,
            "optimizers": [
                GDLS,
                DiagH,
                DiagonalBB,
                PrecSearch,
                RPROP,
                AdaGradLS,
                HyperGDMult,
            ],
            "xlim": 1500,
        },  #
        {
            "problem": LogisticRegression,
            "dataset": "breast-cancer",
            "init": BiasInitializerLogReg,
            "optimizers": [
                GDLS,
                DiagH,
                DiagonalBB,
                PrecSearch,
                RPROP,
                AdaGradLS,
                HyperGDMult,
            ],
            "xlim": 1000,
        },
        {
            "problem": LogisticRegression,  # LinearRegression
            "dataset": "news20.binary",  # "E2006-E2006-tfidf"
            "init": BiasInitializerLogReg,
            "optimizers": [
                DiagH,
                DiagonalBB,
                GDLS,
                RPROP,
                AdaGradLS,
                HyperGDMult,
                PrecSearch,
            ],
            "xlim": 1500,
        },
    ]

    gs = fig.add_gridspec(
        nrows=1,
        ncols=len(to_plot),
        left=0.07,
        right=0.79,
        bottom=0.2,
        top=0.90,
        wspace=0.25,
        hspace=0.2,
    )

    axes = [
        [fig.add_subplot(gs[0, i]) for i in range(len(to_plot))],
    ]

    def plot_exp_on(exps, ax):
        for exp in exps:
            print(exp)

        for exp in exps:
            _data = data["results"][exp.uname()]
            label = precsearch.constants.displayname(exp.opt)
            linestyle = precsearch.constants.linestyles(exp.opt)

            fmin_precomputed = (
                data["problems_data"][exp.prob.uname()]["f"]
                if data["problems_data"][exp.prob.uname()] is not None
                else np.inf
            )

            filtered_results = [data["results"][_.uname()] for _ in exps]

            fmin_observed = np.min([np.min(_["f"]) for _ in filtered_results])
            fmin = np.min([fmin_observed, fmin_precomputed])

            if isinstance(exp.opt, OptPGDLS):
                linestyle["zorder"] = 10
            if isinstance(exp.opt, PrecSearch):
                pass  # linestyle["linewidth"] = 1.0

            if "ngev" not in _data:
                _data["ngev"] = _data["step"]
            if "nfev" not in _data:
                _data["nfev"] = _data["step"] * 0
            if "ndhev" not in _data:
                _data["ndhev"] = _data["step"] * 0
            if "nbacktrack" not in _data:
                _data["nbacktrack"] = _data["step"] * 0

            fvals = _data["f"] - fmin

            fvals = np.array(fvals).clip(max=fvals[0] * 1000)

            ax.plot(
                _data["ngev"] + _data["nfev"],
                fvals,
                label=label,
                **linestyle,
            )
            ax.set_ylim([ax.get_ylim()[0], fvals[0] * 10**1])

    for i in range(len(to_plot)):
        exps_to_plot = [
            exp
            for exp in data["exps"]
            if exp.prob.__class__ == to_plot[i]["problem"]
            and exp.prob.dataset.dataset_name == to_plot[i]["dataset"]
            and exp.prob.init.__class__ == to_plot[i]["init"]
            and (not hasattr(exp.opt, "forward") or exp.opt.forward == 1.1)
            and (
                not hasattr(exp.opt, "set_type")
                or (
                    not exp.opt.set_type == "ellipsoid"
                    or exp.opt.refine_steps == "bfgs"
                )
            )
            and (not isinstance(exp.opt, PrecSearch) or exp.opt.set_type == "ellipsoid")
            and exp.opt.__class__ in to_plot[i]["optimizers"]
        ]
        plot_exp_on(exps_to_plot, axes[0][i])
        axes[0][i].set_title(
            precsearch.constants.displayname(exps_to_plot[0].prob), y=1.0, pad=2
        )
        axes[0][i].set_xlabel("# f/grad evals")

    for row in axes:
        for i, ax in enumerate(row):
            ax.set_yscale("log")
            ax.set_ylim([ax.get_ylim()[1] * 10**-7, ax.get_ylim()[1] * 10])
            ax.set_xlim([-30, to_plot[i]["xlim"]])

    handles, labels = axes[0][-1].get_legend_handles_labels()
    order = [0, 1, 5, 6, 4, 2, 3]

    plt.legend()

    axes[0][-1].legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="lower left",
        bbox_to_anchor=(0.98, 0.0, 1.0, 1.0),
        frameon=False,
    )

    axes[0][0].set_ylabel("Optimality gap")

    pass


if __name__ == "__main__":
    settings(plt)
    fig = plt.figure()
    data = load_data()
    make_figure(fig, postprocess(data))
    filename = f"main_plot.pdf"
    plt.savefig(f"results/{filename}")
    plt.close()
