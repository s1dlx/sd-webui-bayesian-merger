import itertools
import os
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

PathT = os.PathLike

# colormap
NUM_COLORS = 20
COLORMAP_NAME = "vlag"
CMAP = sns.diverging_palette(220, 20, as_cmap=True)
PALETTE = sns.diverging_palette(220, 20, n=NUM_COLORS)
ECOLOR = "k"
LWIDTH = 0.5

# in/out blocks
PAD = 0.02
BLOCK_W = 0.125
BLOCK_H = 0.41
GROUP_V_D = 0.5
GROUP_H_D = 0.25
BLOCK_H_D = 0.1
BLOCK_V_D = 0.1
TEXT_V_D = 0.05

# middle block
MIDDLE_W = 5 * BLOCK_W
MIDDLE_H = 0.5 * BLOCK_H
MIDDLE_H_D = 0.5 * GROUP_H_D
MIDDLE_V_D = 0.75 * GROUP_V_D

# base_alpha
BASE_W = 2 * BLOCK_H
BASE_H = BLOCK_W
BASE_H_V_D = BLOCK_H * 8.5

# text
TEXT_COLOR = "k"
TEXT_SIZE = 10


def get_color(w, palette=PALETTE, num_colors=NUM_COLORS):
    return palette[int(round(w * num_colors)) - 1]


def block_patch(bl, w, h, pad, color, ecolor=ECOLOR, lwidth=LWIDTH):
    return mpatches.FancyBboxPatch(
        bl,
        w,
        h,
        boxstyle=mpatches.BoxStyle("Round", pad=pad),
        facecolor=color,
        edgecolor=ecolor,
        linewidth=lwidth,
    )


def block_text(blt, txt):
    plt.text(
        *blt,
        txt,
        size=TEXT_SIZE,
        ha="center",
        va="top",
        color=TEXT_COLOR,
    )


# in
def in_block(x, y, weights, weight_id, bl, ax):
    bl_y = (GROUP_V_D + BLOCK_H) * (3 - y) + BLOCK_H - x * BLOCK_V_D
    bl_x = x * (BLOCK_W + BLOCK_H_D) + y * GROUP_H_D
    bl = (bl_x, bl_y)
    patch = block_patch(
        bl,
        BLOCK_W,
        BLOCK_H,
        PAD,
        get_color(weights[weight_id]),
    )

    ax.add_artist(patch)
    blt = (bl[0] + BLOCK_W / 2, bl[1] - TEXT_V_D)
    block_text(blt, weight_id)
    weight_id += 1

    return weight_id, bl


def out_block(x, y, weights, weight_id, bl, ax, middle_x0, middle_y0):
    bl_y = (
        middle_y0 + MIDDLE_H + MIDDLE_V_D + (y) * (GROUP_V_D + BLOCK_H) + x * BLOCK_V_D
    )
    bl_x = middle_x0 + MIDDLE_W + x * (BLOCK_W + BLOCK_H_D) + (y) * GROUP_H_D

    bl = (bl_x, bl_y)
    patch = block_patch(
        bl,
        BLOCK_W,
        BLOCK_H,
        PAD,
        get_color(weights[weight_id]),
    )

    ax.add_artist(patch)
    blt = (bl[0] + BLOCK_W / 2, bl[1] - TEXT_V_D)
    block_text(blt, weight_id)
    weight_id += 1

    return weight_id, bl


def draw_unet(
    base_alpha: float,
    weights: List[float],
    model_a: str = "A",
    model_b: str = "B",
    figname: PathT = None,
):
    fig = plt.figure(1, figsize=(7, 5))
    fig.clf()

    ax = fig.add_subplot(111)

    weight_id = 0
    bl = (0.0, 0.0)
    for y, x in itertools.product(range(4), range(3)):
        weight_id, bl = in_block(x, y, weights, weight_id, bl, ax)

    # middle
    bl_x, bl_y = bl
    bl_x += MIDDLE_H_D
    bl_y -= MIDDLE_V_D + MIDDLE_H
    middle_x0 = bl_x
    middle_y0 = bl_y
    bl = (bl_x, bl_y)
    patch = block_patch(
        bl,
        MIDDLE_W,
        MIDDLE_H,
        PAD,
        get_color(weights[weight_id]),
    )
    ax.add_artist(patch)
    blt = (bl[0] + MIDDLE_W / 2, bl[1] - TEXT_V_D)
    block_text(blt, weight_id)
    weight_id += 1

    # base_alpha
    ba_bl = (bl[0] + MIDDLE_W / 2 - BASE_W / 2, bl[1] + MIDDLE_H + BASE_H_V_D)
    patch = block_patch(
        ba_bl,
        BASE_W,
        BASE_H,
        PAD,
        get_color(base_alpha),
    )
    ax.add_artist(patch)
    blt = (bl[0] + MIDDLE_W / 2, ba_bl[1] - TEXT_V_D)
    block_text(blt, "base_alpha")

    # out
    for y, x in itertools.product(range(4), range(3)):
        weight_id, bl = out_block(
            x,
            y,
            weights,
            weight_id,
            bl,
            ax,
            middle_x0,
            middle_y0,
        )

    ax.relim()
    ax.autoscale_view()
    ax.set_axis_off()

    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=PAD * 5, pack_start=True)
    fig.add_axes(cax)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal"
    )
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([model_a, model_b])

    if figname:
        fig.savefig(figname)

    # TODO: weight value inside box?


def maxwhere(li: List[float]) -> Tuple[int, float]:
    m = 0
    mi = -1
    for i, v in enumerate(li):
        if v > m:
            m = v
            mi = i
    return mi, m


def minwhere(li: List[float]) -> Tuple[int, float]:
    m = 10
    mi = -1
    for i, v in enumerate(li):
        if v < m:
            m = v
            mi = i
    return mi, m


def convergence_plot(
    scores: List[float],
    figname: PathT = None,
    minimise=False,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(scores)

    star_i, star_score = minwhere(scores) if minimise else maxwhere(scores)
    plt.plot(star_i, star_score, "or")

    plt.xlabel("iterations")

    if minimise:
        plt.ylabel("loss")
    else:
        plt.ylabel("score")

    sns.despine()

    if figname:
        plt.title(figname.stem)
        print("Saving fig to:", figname)
        plt.savefig(figname)
