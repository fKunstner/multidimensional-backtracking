import warnings
from datetime import datetime
from math import atan2, degrees
from typing import Tuple

import numpy as np
from matplotlib.dates import date2num
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

IN_PER_CM = 0.394


def rgb_to_unit(xs):
    """Convert a list of RGB numbers [0, 255] to a list of unit [0, 1]"""
    return [x / 255.0 for x in xs]


colors = {
    "yellow": rgb_to_unit([221, 170, 51]),
    "red": rgb_to_unit([187, 85, 102]),
    "blue": rgb_to_unit([0, 68, 136]),
    "black": rgb_to_unit([0, 0, 0]),
}


def set_basic_size(fig, width_cm: float, ratio: float, dpi: int = 150):
    fig.set_dpi(dpi)
    figsize_cm = [width_cm, width_cm * ratio]
    figsize_in = [IN_PER_CM * _ for _ in figsize_cm]
    fig.set_size_inches(*figsize_in)


def make_slider(
    fig: Figure,
    coords: Tuple[float, float, float, float],
    label: str,
    start: float,
    end: float,
    init: float,
    step: float,
):
    """Makes a new axis containing a slider at the coordinates

    Coordinates format is ``[left, bottom, width, height]``

    ``label``, ``start``, ``end``, ``init`` and ``step`` are passed to the
    slider constructor, see matplotlib.widgets.Slider

    Returns the slider handle
    """
    return Slider(fig.add_axes(coords), label, start, end, valinit=init, valstep=step)


def ellipse_points(center, mat, scaling: float = 1.0, density: int = 100) -> Tuple:
    """Returns the cartesian coordinates of an ellipse centered at center
    with shape given by the matrix mat.

    Corresponds to the level set ``f(x) = 1`` for
        ``f(x) = .5 * (x-c)^T A (x-c) / scaling``
    """
    circle_grid = np.linspace(0, 2 * np.pi, density)
    inv = np.linalg.inv(mat) * scaling
    xs = inv[0, 0] * np.sin(circle_grid) + inv[0, 1] * np.cos(circle_grid) + center[0]
    ys = inv[1, 1] * np.cos(circle_grid) + inv[1, 0] * np.sin(circle_grid) + center[1]
    return xs, ys


def reset_all_handles(handles_to_reset):
    for handle in handles_to_reset:
        if handle is not None:
            try:
                handle.remove()
            except AttributeError:
                for elem in handle.collections:
                    elem.remove()


def plot_quadratic_contour_and_vectors(ax, InverseH, v1, v2):
    handles = []
    for scaling in [0.5, 1.0, 2.0, 4.0]:
        (_handle,) = ax.plot(
            *ellipse_points(center=[0, 0], mat=InverseH, scaling=scaling),
            color="grey",
        )
        handles.append(_handle)
    (_handle,) = ax.plot([0.0, v1[0]], [0.0, v1[1]], color="k")
    handles.append(_handle)
    (_handle,) = ax.plot([0.0, v2[0]], [0.0, v2[1]], color="k")
    handles.append(_handle)
    return handles


def write_matrix(fig, H, x, y):
    """Writes a text label for a 2d matrix H on the figure at coordinates x,y"""
    matrix_text = f"[{H[0, 0]:.2f}   {H[0, 1]:.2f}]\n [{H[1, 0]:.2f}   {H[1, 1]:.2f}]"

    _handle1 = fig.text(
        x,
        y,
        "H = ",
        horizontalalignment="right",
        verticalalignment="center",
    )
    _handle2 = fig.text(
        x,
        y,
        matrix_text,
        horizontalalignment="left",
        verticalalignment="center",
    )
    return [_handle1, _handle2]


def make_valid_invalid_labels(ax, valid_on_top=True):
    labels = ["Valid", "Invalid"]
    colors = ["black", "white"]
    if not valid_on_top:
        labels.reverse()
        colors.reverse()

    _handle1 = ax.text(
        0.95,
        0.95,
        "Valid",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
    )
    _handle2 = ax.text(
        0.05,
        0.05,
        "Invalid",
        color="white",
        horizontalalignment="left",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )
    return [_handle1, _handle2]


def labelLine(
    line,
    x,
    label=None,
    align=True,
    drop_label=False,
    manual_rotation=0,
    ydiff=0.0,
    **kwargs,
):
    """Label a single matplotlib line at position x

    Source:
    https://github.com/cphyc/matplotlib-label-lines
    https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib

    Parameters
    ----------
    line : matplotlib.lines.Line
       The line holding the label
    x : number
       The location in data unit of the label
    label : string, optional
       The label to set. This is inferred from the line by default
    drop_label : bool, optional
       If True, the label is consumed by the function so that subsequent calls to e.g. legend
       do not use it anymore.
    kwargs : dict, optional
       Optional arguments passed to ax.text
    """
    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    mask = np.isfinite(ydata)
    if mask.sum() == 0:
        raise Exception("The line %s only contains nan!" % line)

    # Find first segment of xdata containing x
    if len(xdata) == 2:
        i = 0
        xa = min(xdata)
        xb = max(xdata)
    else:
        for i, (xa, xb) in enumerate(zip(xdata[:-1], xdata[1:])):
            if min(xa, xb) <= x <= max(xa, xb):
                break
        else:
            raise Exception("x label location is outside data range!")

    def x_to_float(x):
        """Make sure datetime values are properly converted to floats."""
        return date2num(x) if isinstance(x, datetime) else x

    xfa = x_to_float(xa)
    xfb = x_to_float(xb)
    ya = ydata[i]
    yb = ydata[i + 1]
    y = ya + (yb - ya) * (x_to_float(x) - xfa) / (xfb - xfa)

    if not (np.isfinite(ya) and np.isfinite(yb)):
        warnings.warn(
            (
                "%s could not be annotated due to `nans` values. "
                "Consider using another location via the `x` argument."
            )
            % line,
            UserWarning,
        )
        return

    if not label:
        label = line.get_label()

    if drop_label:
        line.set_label(None)

    if align:
        # Compute the slope and label rotation
        screen_dx, screen_dy = ax.transData.transform(
            (xfa, ya)
        ) - ax.transData.transform((xfb, yb))
        rotation = (degrees(atan2(screen_dy, screen_dx)) + 90) % 180 - 90
    else:
        rotation = manual_rotation

    # Set a bunch of keyword arguments
    if "color" not in kwargs:
        kwargs["color"] = line.get_color()

    if ("horizontalalignment" not in kwargs) and ("ha" not in kwargs):
        kwargs["ha"] = "center"

    if ("verticalalignment" not in kwargs) and ("va" not in kwargs):
        kwargs["va"] = "center"

    if "backgroundcolor" not in kwargs:
        kwargs["backgroundcolor"] = ax.get_facecolor()

    if "clip_on" not in kwargs:
        kwargs["clip_on"] = True

    if "zorder" not in kwargs:
        kwargs["zorder"] = 2.5

    ax.text(x, y + ydiff, label, rotation=rotation, **kwargs)


def subsample_idx(length, n, log=False):
    """Returns a n-subset of [0,length-1]"""
    if log:
        log_grid = np.logspace(start=0, stop=np.log10(length - 1), num=n - 1)
        idx = [0] + list(log_grid.astype(int))
    else:
        lin_grid = np.linspace(start=0, stop=length - 1, num=n)
        idx = list(lin_grid.astype(int))
    idx = sorted(list(set(idx)))
    return idx


def subsample(xs, n=100, log=False):
    aslist = list(xs)
    return [aslist[i] for i in subsample_idx(len(aslist), n=n, log=False)]
