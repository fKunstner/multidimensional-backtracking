import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from liveplot import module_loader

import precsearch
import precsearch.constants
import precsearch.optimizers
import precsearch.plotting
import precsearch.plotting_style
from precsearch import config
from precsearch.experiments.constants import load_solution
from precsearch.initializations import (
    BiasInitializerLinReg,
    BiasInitializerLogReg,
    GaussianInitializer,
)
from precsearch.optimizers.optimizers import (
    LBFGS,
    AdaGrad,
    AdaGradNorm,
    HyperGDAdd,
    OptPGDLS,
    PrecSearch,
    RMSProp,
)


def load_data(file=None, dataset=None, init=None):
    importlib.reload(precsearch)

    if file is None:
        file = "exp_big_logreg.py"
        file = "exp_small_logreg.py"
        file = "exp_small_linreg.py"

    if dataset is None:
        if file == "exp_small_linreg.py":
            dataset = "mg"
            dataset = "california-housing"
            dataset = "concrete"
            dataset = "power-plant"
            dataset = "cpusmall"

        if file == "exp_small_logreg.py":
            dataset = "diabetes"
            dataset = "ionosphere"
            dataset = "breast-cancer"
            dataset = "heart"
            dataset = "australian"

        if file == "exp_big_logreg.py":
            dataset = "cifar10"
            dataset = "rcv1.binary"
            dataset = "news20.binary"

    if init is None:
        init = BiasInitializerLogReg
        init = BiasInitializerLinReg
        init = GaussianInitializer

    ellipsoid_refine = "bfgs"
    filter_only_sane = True

    def select(exp):
        if exp.prob.dataset.dataset_name != dataset:
            return False
        if exp.prob.init.__class__ != init:
            return False
        if isinstance(exp.opt, PrecSearch):
            if (
                exp.opt.set_type == "ellipsoid"
                and exp.opt.refine_steps != ellipsoid_refine
            ):
                return False
            if exp.opt.set_type == "box":
                return False
        if hasattr(exp.opt, "forward"):
            if exp.opt.forward == 2.0:
                return False
        if filter_only_sane:
            if hasattr(exp.opt, "forward") and exp.opt.forward == 1.0:
                return False
        if filter_only_sane:
            if isinstance(exp.opt, HyperGDAdd):
                return False
            if isinstance(exp.opt, AdaGradNorm):
                return False
            if isinstance(exp.opt, AdaGrad):
                return False
            if isinstance(exp.opt, RMSProp):
                return False
        if isinstance(exp.opt, LBFGS):
            return False
        return True

    foobar = module_loader._import_module(Path(file))

    def print_thingies(name, thingies):
        print(name + ":")
        for thing in sorted(list(set([str(thing) for thing in thingies]))):
            print("    ", thing)

    print()
    print_thingies("datasets", [exp.prob.dataset for exp in foobar.experiments])
    print_thingies("inits", [exp.prob.init for exp in foobar.experiments])
    print_thingies("opts", [exp.opt for exp in foobar.experiments])
    print()
    filtered_exp = [exp for exp in foobar.experiments if select(exp)]
    print(
        f"Selected [{dataset}, {init}, {ellipsoid_refine}] ({len(filtered_exp)} runs)"
    )
    print()

    return {"exps": filtered_exp}


def postprocess(data):
    data["results"] = {exp.uname(): exp.load() for exp in data["exps"]}

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
            rel_width=1.0, nrows=1, ncols=3, height_to_width_ratio=0.7
        )
    )


def make_figure(fig, data):
    importlib.reload(precsearch.constants)

    # axes = [fig.add_subplot(111)]
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.07,
        right=0.98,
        bottom=0.22,
        top=0.85,
        wspace=0.15,
    )

    fig.set_dpi(250)

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    exps_ordered = data["exps"]

    for exp in exps_ordered:
        _data = data["results"][exp.uname()]
        label = precsearch.constants.displayname(exp.opt)
        linestyle = precsearch.constants.linestyles(exp.opt)

        fmin_precomputed = (
            data["problems_data"][exp.prob.uname()]["f"]
            if data["problems_data"][exp.prob.uname()] is not None
            else np.inf
        )
        fmin_observed = np.min([np.min(_["f"]) for k, _ in data["results"].items()])
        fmin = np.min([fmin_observed, fmin_precomputed])
        # fmin = 0

        linestyle["linewidth"] = 2

        if isinstance(exp.opt, OptPGDLS):
            linestyle["zorder"] = 10
        if isinstance(exp.opt, PrecSearch):
            pass  # linestyle["linewidth"] = 1.0

        if "ngev" not in _data:
            _data["ngev"] = _data["step"]
        if "nfev" not in _data:
            _data["nfev"] = _data["step"]
        if "ndhev" not in _data:
            _data["ndhev"] = _data["step"] * 0
        if "nbacktrack" not in _data:
            _data["nbacktrack"] = _data["step"] * 0

        fvals = _data["f"] - fmin
        fvals = np.array(fvals).clip(max=fvals[0] * 1000)

        axes[0].plot(
            _data["ngev"] + _data["nfev"] + _data["ndhev"],
            fvals,
            label=label,
            **linestyle,
        )

        actual_steps = [
            iteration - nbacktrack + _data["nbacktrack"][0]
            for iteration, nbacktrack in zip(_data["step"], _data["nbacktrack"])
        ]

        axes[1].plot(actual_steps, fvals, label=label, **linestyle)

        for ax in axes:
            ax.set_ylim([ax.get_ylim()[0], fvals[0] * 10**1])

    for ax in [axes[0]]:
        ax.set_yscale("log")
    axes[1].set_yscale("log")

    oracle_max = {
        "rcv1.binary": 1500,
        "news20.binary": 1500,
        "breast-cancer": 1500,
        "heart": 1500,
        "australian": 1500,
        "diabetes": 1500,
        "ionosphere": 1500,
        "california-housing": 4000,
        "concrete": 4000,
        "power-plant": 4000,
        "cpusmall": 4000,
        "mg": 4000,
    }
    step_max = {
        "rcv1.binary": 500,
        "news20.binary": 500,
        "breast-cancer": 500,
        "heart": 500,
        "australian": 500,
        "diabetes": 500,
        "ionosphere": 500,
        "california-housing": 1500,
        "concrete": 1500,
        "power-plant": 1500,
        "cpusmall": 1500,
        "mg": 1500,
    }

    precsearch_exps = [exp for exp in exps_ordered if isinstance(exp.opt, PrecSearch)]
    precsearch_expname = precsearch_exps[0].uname()
    dsname = precsearch_exps[0].prob.dataset.dataset_name
    is_big_dataset = dsname in ["rcv1.binary", "news20.binary"]
    results_as_array = np.array(data["results"][precsearch_expname]["f"])
    min_precsearch = np.min(results_as_array[: step_max[dsname] * 2])

    for ax in axes:
        if is_big_dataset:
            ymin = (min_precsearch - fmin) / 1000
        else:
            ymin = (min_precsearch - fmin) / 100

        ax.set_ylim([ymin, ax.get_ylim()[1] * 10])

    axes[0].set_xlim([-20, oracle_max[dsname]])
    axes[1].set_xlim([-20, step_max[dsname]])

    axes[0].set_xlabel("Oracle calls (Function + Gradient eval.)")
    axes[1].set_xlabel("Iterations (Changes in $x$)")
    axes[0].set_ylabel("Optimality gap")
    fig.suptitle(precsearch.constants.displayname(exps_ordered[0].prob.dataset), y=1.00)

    # axes[1].legend(
    #     loc="lower left", bbox_to_anchor=(1.1, -0.3, 1.0, 1.0), frameon=False
    # )

    pass


if __name__ == "__main__":
    settings(plt)

    configs_logreg = {
        "exp_big_logreg.py": ["rcv1.binary", "news20.binary"],
        "exp_small_logreg.py": [
            "breast-cancer",
            "heart",
            "australian",
            "diabetes",
            "ionosphere",
        ],
    }
    configs_linreg = {
        "exp_small_linreg.py": [
            "california-housing",
            "concrete",
            "power-plant",
            "cpusmall",
            "mg",
        ],
    }

    def makeplot(file, dsname, init):
        fig = plt.figure()
        data = load_data(file=file, dataset=dsname, init=init)
        make_figure(fig, postprocess(data))
        filename = f"{file.replace('.py', '')}_{init.__name__}_{dsname}.pdf"
        plt.savefig(f"results/{filename}")
        print("Saved", dsname)
        plt.close()

    for file, datasets in configs_logreg.items():
        for ds in datasets:
            for init in [BiasInitializerLogReg, GaussianInitializer]:
                makeplot(file, ds, init)
    for file, datasets in configs_linreg.items():
        for ds in datasets:
            for init in [BiasInitializerLinReg, GaussianInitializer]:
                makeplot(file, ds, init)
