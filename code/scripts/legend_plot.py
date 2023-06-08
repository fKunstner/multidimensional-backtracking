import importlib

import matplotlib
import matplotlib.pyplot as plt

import precsearch
import precsearch.constants
import precsearch.optimizers
import precsearch.plotting
import precsearch.plotting_style
from precsearch.optimizers.optimizers import (
    GDLS,
    RPROP,
    AdaGradLS,
    DiagH,
    DiagonalBB,
    HyperGDMult,
    PrecSearch,
)


def load_data():
    return


def postprocess(data):
    return


def settings(plt):
    import importlib

    importlib.reload(precsearch.plotting_style)

    plt.rcParams.update(
        precsearch.plotting_style.neurips_config(
            rel_width=1.0, nrows=1, ncols=3, height_to_width_ratio=0.3
        )
    )


def make_figure(fig, data):
    importlib.reload(precsearch.constants)

    optims = [
        GDLS(),
        PrecSearch(set_type="ellipsoid"),
        None,
        DiagH(),
        DiagonalBB(),
        None,
        AdaGradLS(),
        RPROP(),
        HyperGDMult(),
    ]

    lines = []
    for opt in optims:
        if opt is None:
            linestyle = {"linewidth": 0}
            label = ""
        else:
            linestyle = precsearch.constants.linestyles(opt)
            label = precsearch.constants.displayname(opt)
        lines.append(matplotlib.lines.Line2D([], [], **linestyle, label=label))

    leg = fig.legend(
        handles=lines,
        loc="center",
        ncol=3,
    )
    fig.add_artist(leg)


if __name__ == "__main__":
    settings(plt)

    fig = plt.figure()
    make_figure(fig, data=None)
    filename = f"legend.pdf"
    plt.savefig(f"results/{filename}")
    plt.close()
