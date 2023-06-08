import precsearch.plotting
from precsearch.datasets import Dataset
from precsearch.optimizers.optimizers import (
    GD,
    GDLS,
    LBFGS,
    RPROP,
    AdaGradLS,
    DiagH,
    DiagonalBB,
    HyperGDMult,
    OptGD,
    Optimizer,
    OptPGD,
    OptPGDLS,
    PrecSearch,
)
from precsearch.problems import LinearRegression, LogisticRegression, Problem


def displayname(thing):
    label = "N/A"

    if isinstance(thing, Dataset):
        if "E2006" in thing.dataset_name:
            label = "E2006"
        elif "news20" in thing.dataset_name:
            label = "News20"
        elif "rcv1" in thing.dataset_name:
            label = "RCV1"
        elif "cpusmall" in thing.dataset_name:
            label = "cpusmall"
        elif "Breast-cancer" in thing.dataset_name:
            label = "breast-cancer"
        else:
            label = thing.dataset_name.capitalize()
    elif isinstance(thing, Problem):
        if isinstance(thing, LinearRegression):
            label = (
                f"Linear - {displayname(thing.dataset)}"
                # f"($\\lambda = {thing.regularization}$)"
            )
        elif isinstance(thing, LogisticRegression):
            label = (
                f"Logistic - {displayname(thing.dataset)}"
                # f"($\\lambda = {thing.regularization}$)"
            )
    elif isinstance(thing, Optimizer):
        label = thing.__class__.__name__
        if isinstance(thing, OptGD) or isinstance(thing, GD):
            label = "GD"
        if isinstance(thing, OptPGD):
            # label = "Opt. Prec. GD"
            label = "$\mathbf{P}_*$GD"
        if isinstance(thing, DiagonalBB):
            label = "Diag. BB+NMLS"
        if isinstance(thing, DiagH):
            label = "Diag. Hessian+LS"
        if isinstance(thing, AdaGradLS):
            label = "Diag. AdaGrad+LS"
        if isinstance(thing, HyperGDMult):
            label = "GD-HD (mult.)"
        if isinstance(thing, GDLS):
            if thing.backtrack == 1.0 and thing.forward == 1.0:
                label = "GD"
            else:
                label = "GD+LS"
        if isinstance(thing, PrecSearch):
            label = thing.set_type.capitalize() + " MB"
        if isinstance(thing, OptPGDLS):
            if thing.backtrack == 1.0 and thing.forward == 1.0:
                label = "$\mathbf{P}_*$GD"
            else:
                label = "$\mathbf{P}_*$GD + LS"

    return label


def linestyles(opt):
    linestyle = {"linewidth": 1.5}
    if isinstance(opt, PrecSearch):
        linestyle["color"] = precsearch.plotting.colors["red"]
        linestyle["zorder"] = 40
        if opt.set_type == "box":
            linestyle["marker"] = "s"
            linestyle["markeredgecolor"] = "black"
            linestyle["markeredgewidth"] = 1
            linestyle["markevery"] = 0.4
        elif opt.set_type == "simplex":
            linestyle["marker"] = "v"
            linestyle["markeredgecolor"] = "black"
            linestyle["markeredgewidth"] = 1
            linestyle["markevery"] = 0.35
        elif opt.set_type == "ellipsoid":
            linestyle["marker"] = "o"
            linestyle["markeredgecolor"] = "black"
            linestyle["markeredgewidth"] = 1
            linestyle["markevery"] = 0.3
    elif isinstance(opt, LBFGS):
        linestyle["color"] = precsearch.plotting.colors["blue"]
        linestyle["linestyle"] = "--"
    elif isinstance(opt, GD) or isinstance(opt, OptGD):
        linestyle["color"] = precsearch.plotting.colors["black"]
        linestyle["linestyle"] = "--"
    elif isinstance(opt, GDLS):
        linestyle["color"] = precsearch.plotting.colors["black"]
        linestyle["zorder"] = 30
    elif isinstance(opt, OptPGD):
        linestyle["linestyle"] = "-"
        linestyle["color"] = precsearch.plotting.colors["blue"]
    elif isinstance(opt, OptPGDLS):
        linestyle["color"] = precsearch.plotting.colors["blue"]
        linestyle["linestyle"] = "-"
    elif isinstance(opt, DiagH):
        linestyle["color"] = precsearch.plotting.colors["blue"]
        linestyle["linestyle"] = "--"
        linestyle["zorder"] = 20
    elif isinstance(opt, AdaGradLS):
        linestyle["color"] = precsearch.plotting.colors["yellow"]
        linestyle["linestyle"] = "--"
    elif isinstance(opt, HyperGDMult):
        linestyle["color"] = precsearch.plotting.colors["yellow"]
        linestyle["linewidth"] = 1
        linestyle["linestyle"] = "-."
        linestyle["zorder"] = 1
    elif isinstance(opt, DiagonalBB):
        linestyle["color"] = precsearch.plotting.colors["blue"]
        linestyle["linestyle"] = "-"
        linestyle["linewidth"] = 1
        linestyle["zorder"] = 1
    elif isinstance(opt, RPROP):
        linestyle["color"] = precsearch.plotting.colors["yellow"]
        linestyle["linestyle"] = "-"
        linestyle["zorder"] = 5

    return linestyle
