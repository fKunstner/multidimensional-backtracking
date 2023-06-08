# Magic constants
COLOR_GREY = "#808080"
COLOR_BLACK = "#000000"

_stroke_width = 0.5
_xtick_width = 0.8
_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0


def base_font(*, family="sans-serif"):
    # ptmx replacement
    fontset = "stix" if family == "serif" else "stixsans"
    return {
        "text.usetex": False,
        "font.sans-serif": ["TeX Gyre Heros"],
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": fontset,
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.family": family,
    }


fontsizes = {
    "normal": 9,
    "small": 7,
    "tiny": 6,
}


def base_fontsize(*, base=10):
    fontsizes = {
        "normal": base - 1,
        "small": base - 3,
        "tiny": base - 4,
    }

    return {
        "font.size": fontsizes["normal"],
        "axes.titlesize": fontsizes["normal"],
        "axes.labelsize": fontsizes["small"],
        "legend.fontsize": fontsizes["small"],
        "xtick.labelsize": fontsizes["tiny"],
        "ytick.labelsize": fontsizes["tiny"],
    }


def base_layout(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=2,
    constrained_layout=False,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
    base_width_in=5.5,
):
    width_in = base_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    figsize = (width_in, height_in)

    return {
        "figure.dpi": 150,
        "figure.figsize": figsize,
        "figure.constrained_layout.use": constrained_layout,
        "figure.autolayout": tight_layout,
        # Padding around axes objects. Float representing
        "figure.constrained_layout.h_pad": 1 / 72,
        # inches. Default is 3/72 inches (3 points)
        "figure.constrained_layout.w_pad": 1 / 72,
        # Space between subplot groups. Float representing
        "figure.constrained_layout.hspace": 0.00,
        # a fraction of the subplot widths being separated.
        "figure.constrained_layout.wspace": 0.00,
    }


def base_style():
    grid_color = COLOR_GREY
    text_color = COLOR_BLACK
    return {
        "text.color": text_color,
        "axes.labelcolor": text_color,
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": grid_color,
        "axes.linewidth": _stroke_width,
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "grid.color": grid_color,
        "grid.linewidth": _stroke_width,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def smaller_style():
    return {
        "axes.labelpad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "ytick.major.pad": 1,
        "xtick.major.pad": 1,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "axes.titlepad": 3,
    }


def neurips_config(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=4,
    family="sans-serif",
    height_to_width_ratio=_GOLDEN_RATIO,
):
    font_config = base_font(family=family)
    fonsize_config = base_fontsize(base=10)
    layout_config = base_layout(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
        base_width_in=5.5,
    )
    style_config = smaller_style()
    return {**font_config, **fonsize_config, **layout_config, **style_config}
