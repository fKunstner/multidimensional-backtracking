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
from precsearch.experiments.constants import load_solution
from precsearch.experiments.expdefs import small_linear_regression_datasets
from precsearch.optimizers.optimizers import OptPGD, OptPGDLS, PrecSearch


def load_data(dataset="cpusmall"):
    importlib.reload(precsearch)

    foobar = module_loader._import_module(Path("experiments_fig1.py"))
    filtered_exp = [
        exp for exp in foobar.experiments if exp.prob.dataset.dataset_name == dataset
    ]

    return {"exps": filtered_exp}


def postprocess(data):
    data["results"] = {exp.uname(): exp.load() for exp in data["exps"]}

    unique_problems = {exp.prob.uname(): exp.prob for exp in data["exps"]}
    data["problems_data"] = {
        name: load_solution(problem) for name, problem in unique_problems.items()
    }
    return data


def settings(plt):
    import importlib

    importlib.reload(precsearch.plotting_style)
    plt.rcParams.update(
        precsearch.plotting_style.neurips_config(
            rel_width=1.0,
            nrows=1,
            ncols=3,
        )
    )


def make_figure(fig, data):
    importlib.reload(precsearch.constants)

    fig.set_dpi(250)

    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        left=0.045,
        right=0.96,
        bottom=0.23,
        top=0.92,
        wspace=0.6,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    reorder_idx = [1, 2, 4, 5]

    exps_ordered = [data["exps"][i] for i in reorder_idx]

    for exp in exps_ordered:
        _data = data["results"][exp.uname()]
        label = precsearch.constants.displayname(exp.opt)
        linestyle = precsearch.constants.linestyles(exp.opt)

        fmin_observed = np.min(
            [np.min(data["results"][_.uname()]["f"]) for _ in exps_ordered]
        )
        fmin_precomputed = data["problems_data"][exp.prob.uname()]["f"]
        fmin = np.min([fmin_observed, fmin_precomputed])

        if isinstance(exp.opt, OptPGDLS):
            linestyle["zorder"] = 10
        if isinstance(exp.opt, PrecSearch):
            pass  # linestyle["linewidth"] = 1.0

        fvals = _data["f"] - fmin
        axes[0].plot(_data["ngev"], fvals, label=label, **linestyle)
        ##
        if True:
            if "stepsizes" in _data:
                if isinstance(exp.opt, PrecSearch) and exp.opt.set_type == "box":
                    if exp.opt.set_type == "box":
                        timesteps = [0, 30, 45, 65]
                    else:
                        timesteps = [0, 50, 100, 200]

                    linestyle["marker"] = ""
                    stepsizes = np.array(
                        _data["stepsizes"]
                        .str.replace(" ", ", ", regex=False)
                        .str.replace(", , ", ", ", regex=False)
                        .str.replace(", , ", ", ", regex=False)
                        .str.replace(", , ", ", ", regex=False)
                        .str.replace("[, ", "[", regex=False)
                        .apply(eval)
                        .tolist()
                    )
                    dims = range(1, len(stepsizes[0, :]) + 1)
                    dims = list(reversed(dims))
                    for t in timesteps:
                        axes[1].plot(dims, stepsizes[t, :], **linestyle)
                        axes[1].text(
                            dims[0] + 0.5,
                            stepsizes[t, :][0],
                            f"T={t}",
                            size="x-small",
                        )
                if isinstance(exp.opt, OptPGDLS) or isinstance(exp.opt, OptPGD):
                    stepsizes = np.array(
                        _data["stepsizes"]
                        .str.replace(" ", ", ", regex=False)
                        .str.replace(", , ", ", ", regex=False)
                        .str.replace(", , ", ", ", regex=False)
                        .str.replace(", , ", ", ", regex=False)
                        .str.replace("[, ", "[", regex=False)
                        .apply(eval)
                        .tolist()
                    )
                    dims = range(1, len(stepsizes[0, :]) + 1)
                    dims = list(reversed(dims))

                    linestyle["linestyle"] = ""
                    linestyle["marker"] = "."
                    linestyle["zorder"] = 100

                    axes[1].plot(dims, (stepsizes[300, :]), **linestyle)
                    dims = list(reversed(dims))
                    axes[1].set_xlim([dims[0] - 0.5, dims[-1] + 0.5])
                    axes[1].set_xticks(dims)
                    axes[1].set_xticklabels(
                        [str(x) if i % 2 == 0 else "" for i, x in enumerate(dims)]
                    )

    for ax in [axes[0]]:
        ax.set_yscale("log")
    axes[1].set_yscale("log")

    axes[1].set_xlabel("Coordinate")
    axes[0].set_ylim([fvals[0] * 10**-5, fvals[0] * 10])
    axes[0].set_xlim([0, 1000])
    axes[1].set_ylim([axes[1].get_ylim()[0] * 10**-4, axes[1].get_ylim()[1] * 10])

    axes[0].set_title(precsearch.constants.displayname(exp.prob))
    axes[0].set_xlabel("Gradient evaluations")
    axes[0].set_title("Optimality gap", pad=0, y=1.0)
    axes[1].set_title("Per-coordinate stepsizes", pad=0, y=1.0)

    axes[0].legend(loc="lower left", bbox_to_anchor=(1.0, 0.0, 1.0, 1.0), frameon=False)


if __name__ == "__main__":
    settings(plt)

    def plot_and_save(dataset):
        fig = plt.figure()
        data = load_data(dataset)
        make_figure(fig, postprocess(data))
        plt.savefig(f"results/toy_linreg_{dataset}.pdf")
        print("Saved", dataset)
        plt.close()

    for dataset in small_linear_regression_datasets:
        plot_and_save(dataset)
