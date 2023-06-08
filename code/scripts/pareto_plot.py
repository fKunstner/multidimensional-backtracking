import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import linalg

import precsearch
import precsearch.plotting as fplt
from precsearch import plotting_style


def settings(plt):
    import importlib

    importlib.reload(plotting_style)
    plt.rcParams.update(
        plotting_style.neurips_config(
            rel_width=0.75, nrows=1, ncols=2, height_to_width_ratio=1.0
        )
    )


def make_figure(
    fig,
    data=None,
    H=np.array(
        [
            [0.5, 0.1],
            [0.1, 1.0],
        ]
    ),
    z=0.96,
    lambda1=0.5,
    lambda2=1.0,
):
    import importlib

    importlib.reload(fplt)

    axes = [
        fig.add_subplot(121),
        fig.add_subplot(122),
    ]

    if H is None:
        theta = 180 * np.arccos(z) / np.pi
        nv1 = np.array([np.cos(np.pi * theta / 180), np.sin(np.pi * theta / 180)])
        nv2 = np.array(
            [np.cos(np.pi * (theta + 90) / 180), np.sin(np.pi * (theta + 90) / 180)]
        )
        H = np.outer(nv1, nv1) * lambda1 + np.outer(nv2, nv2) * lambda2
    else:
        lambda1, lambda2 = np.linalg.eigvalsh(H)
    print(H)

    out = precsearch.optimal_preconditioner(H, verbose=True)
    P = out**2

    print("best stepsizes: ", P)
    print("best stepsize: ", 1 / np.max(np.linalg.eigvalsh(H)))

    def f(x1, x2):
        v = np.array([x1, x2])
        return 0.5 * np.inner(v, H @ v)

    N = 100
    stepsize_xs = np.linspace(-2.0, 2.0, N)
    stepsize_ys = np.linspace(-2.0, 2.0, N)
    Z = np.zeros((N, N))

    for i, x in enumerate(stepsize_xs):
        for j, y in enumerate(stepsize_ys):
            Z[i, j] = f(x, y)
    X, Y = np.meshgrid(stepsize_xs, stepsize_ys)

    axes[0].contourf(X, Y, np.sqrt(Z.T), levels=5, cmap=plt.cm.bone)

    axes[0].set_xlim([-2, 2])
    axes[0].set_ylim([-2, 2])
    axes[0].set_xticks([-2, -1, 0, 1, 2])
    axes[0].set_yticks([-2, -1, 0, 1, 2])

    ##
    # Axes 1

    N = 100000

    eps = 1e-12

    def max_inverse_stepsize_in_y_given_inverse_stepsize_in_x(x):
        return H[1, 1] - H[0, 1] ** 2 / (H[0, 0] - x)

    inverse_xs = np.linspace(eps, 100, N)
    inverse_ys = [
        max_inverse_stepsize_in_y_given_inverse_stepsize_in_x(x) for x in inverse_xs
    ]
    filter = [
        (x, y) for x, y in zip(inverse_xs, inverse_ys) if x > H[0, 0] and y > H[1, 1]
    ]
    stepsize_xs = [1 / xy[0] for xy in filter]
    stepsize_ys = [1 / xy[1] for xy in filter]

    axes[1].plot(stepsize_xs, stepsize_ys, color="k")

    max_per_coord_stepsize = np.max([np.max(stepsize_xs), np.max(stepsize_ys)])
    xymax = max_per_coord_stepsize + 0.251

    exts_xs = [xymax] + stepsize_xs + [0.0]
    exts_ys = [0.0] + stepsize_ys + [stepsize_ys[-1]]

    axes[1].fill_between(
        exts_xs, exts_ys, [xymax for _ in exts_xs], color="black", alpha=0.1
    )

    L = np.max([lambda1, lambda2])
    axes[1].plot([1 / L], [1 / L], "o", color="k", label="1/L")
    axes[1].legend()

    axes[1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(6))

    axes[0].set_title("Function")
    axes[0].set_xlabel("$\mathbf{x}_1$")
    axes[0].set_ylabel("$\mathbf{x}_2$")
    axes[1].set_title("Valid Preconditioners")
    axes[1].set_xlim([0, xymax])
    axes[1].set_ylim([0, xymax])
    axes[1].set_xlabel("$\mathbf{p}_1$")
    axes[1].set_ylabel("$\mathbf{p}_2$")

    fig.tight_layout()


if __name__ == "__main__":
    settings(plt)

    fig = plt.figure()
    make_figure(fig, z=0.71, lambda1=0.001, lambda2=1.0)
    plt.savefig(f"results/pareto_1.pdf")
    plt.close()

    fig = plt.figure()
    make_figure(fig, z=0.95, lambda1=0.5, lambda2=1.0)
    plt.savefig(f"results/pareto_2.pdf")
    plt.close()

    H1 = np.array([[0.5, 0.1], [0.1, 1.0]])

    fig = plt.figure()
    make_figure(fig, H=H1)
    plt.savefig(f"results/pareto_3.pdf")
    plt.close()

    H1 = np.array([[0.5, -0.5], [-0.5, 0.5]])

    fig = plt.figure()
    make_figure(fig, H=H1)
    plt.savefig(f"results/pareto_4.pdf")
    plt.close()

    H1 = np.array([[1, -1], [-1, 1]])

    fig = plt.figure()
    make_figure(fig, H=H1)
    plt.savefig(f"results/pareto_5.pdf")
    plt.close()
