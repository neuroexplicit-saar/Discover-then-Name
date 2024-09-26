from copy import copy
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import os

from .drawing_functions import set_defaults
from .vis_utils import latex_format, savefig


class Canvas:
    DEFAULTS = dict(save_path="", use_latex=False, fontsize=24)

    def __init__(self, fig, ax, extent=None, **defaults):
        set_defaults(defaults, **Canvas.DEFAULTS)
        self.defaults = defaults
        self.fig = fig
        if extent is None:
            extent = [*ax.get_xlim(), ax.get_ylim()]

        self.extent = np.array(extent).flatten()
        self.ax = ax

    def do(self, draw_func, *args, **kwargs):
        draw_func(self, *args, **kwargs)
        return self

    @staticmethod
    def make(width=100, height=100, left=0, top=0, right=0, bottom=0, scale=40):
        fig, ax = plt.subplots(
            1, figsize=np.array([width, height]) * scale / (height + width)
        )
        ax.set_xlim(
            -0.02 * width, width + 0.02 * width
        )  # offset for showing borders correctly
        ax.set_ylim(-0.02 * height, height + 0.02 * height)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
        return Canvas(fig, ax, extent=(left, width - right, bottom, height - top))

    def reset(self):
        self.extent = [*self.ax.get_xlim(), *self.ax.get_ylim()]
        return self

    def zoom(self, new_extent):
        class Zoom:
            prev_extent = None

            def __init__(self, canvas, extent):
                self.prev_extent = canvas.extent
                self.new_extent = extent
                self.canvas = canvas

            def __enter__(self):
                x0, x1, y0, y1 = self.canvas.extent
                xdiff, ydiff = x1 - x0, y1 - y0
                x0_z, x1_z, y0_z, y1_z = self.new_extent
                x0, x1 = x0 + x0_z * xdiff, x0 + x1_z * xdiff
                y0, y1 = y0 + y0_z * ydiff, y0 + y1_z * ydiff
                self.canvas.extent = x0, x1, y0, y1
                return self.canvas

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.canvas.extent = self.prev_extent

        return Zoom(self, new_extent)

    def save(self, sub_path):
        save_path = self.defaults["save_path"]
        # os.makedirs(join(save_path, sub_path), exist_ok=True)
        savefig(self.fig, join(save_path, sub_path))

    def select(self, new_extent):
        class Zoom:
            prev_extent = None

            def __init__(self, canvas, extent):
                self.prev_extent = canvas.extent
                self.new_extent = extent
                self.canvas = canvas

            def __enter__(self):
                self.canvas.extent = self.new_extent
                return self.canvas

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.canvas.extent = self.prev_extent

        return Zoom(self, new_extent)

    def horizontal_split(self, n, pad):
        extents = []
        tmp_v = np.diff(self.extent[:2])[0] + pad

        width_w_pad = tmp_v / n
        single_width = width_w_pad - pad

        extents = [
            [
                self.extent[0] + i * width_w_pad,
                self.extent[0] + i * width_w_pad + single_width,
                *self.extent[2:],
            ]
            for i in range(n)
        ]
        return [
            Canvas(self.fig, self.ax, extent=extent, **self.defaults)
            for extent in extents
        ]

    def vertical_split(self, n, pad):
        extents = []
        tmp_v = np.diff(self.extent[2:])[0] + pad

        width_w_pad = tmp_v / n
        single_width = width_w_pad - pad

        extents = [
            [
                *self.extent[:2],
                self.extent[2] + i * width_w_pad,
                self.extent[2] + i * width_w_pad + single_width,
            ]
            for i in range(n)
        ]
        return [
            Canvas(self.fig, self.ax, extent=extent, **self.defaults)
            for extent in extents
        ][::-1]

    def even_grid(self, rows, cols, pad=0):
        return np.array(
            [ax.horizontal_split(cols, pad)
             for ax in self.vertical_split(rows, pad)]
        )

    def imshow(self, *args, **kwargs):
        return self.ax.imshow(*args, extent=list(self.extent), **kwargs)

    def show_frame(self, **kwargs):
        set_defaults(
            kwargs,
            alpha=0.5,
            edgecolor=kwargs.pop("color", "black"),
            lw=0.5,
            facecolor="none",
        )

        frame = patches.Rectangle(
            self.extent[::2],
            np.diff(self.extent[:2])[0],
            np.diff(self.extent[2:])[0],
            **kwargs
        )
        self.ax.add_patch(frame)
        return self

    def set_xticks(self, *args, **kwargs):
        self.ax.set_xticks(*args, **kwargs)
        return self

    def set_yticks(self, *args, **kwargs):
        self.ax.set_yticks(*args, **kwargs)
        return self

    def set_ylabel(self, s, rotation=90, x_off=-0.05, y_pos=0.5, **kwargs):
        set_defaults(kwargs, ha="right")
        return self.make_text(x_off, y_pos, s, rotation=rotation, **kwargs)

    def make_text(self, x_pos, y_pos, s, **kwargs):
        if not s:
            return self
        x, x2, y, y2 = self.extent
        w, h = x2 - x, y2 - y
        set_defaults(
            kwargs,
            fontsize=self.defaults["fontsize"],
            fontweight="bold",
            va="center",
            ha="center",
        )
        self.ax.text(
            x + x_pos * w,
            y + y_pos * h,
            latex_format(
                s,
                use_latex=self.defaults["use_latex"],
                bf=kwargs["fontweight"] == "bold",
            ),
            **kwargs
        )
        return self

    def set_title(self, s, where="above", y_off=0.05, x_pos=0.5, **kwargs):
        set_defaults(kwargs, va="bottom" if where == "above" else "top")
        return self.make_text(
            x_pos, (1 + y_off) if where == "above" else -y_off, s, **kwargs
        )

    def __repr__(self):
        return ""
