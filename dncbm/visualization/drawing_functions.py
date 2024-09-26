import numpy as np
import torch.nn.functional as F
from matplotlib import colors as plt_colors
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from torch.autograd import Variable
import torch

def to_numpy(tensor: "Union[torch.Tensor, np.ndarray]") -> "np.ndarray":
    """
    Convert torch tensor to NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Tensor to convert.

    Returns
    -------
    np.ndarray
        Converted tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


def to_numpy_img(tensor):
    """
    Converting tensor to numpy image. Expects a tensor of at most 3 dimensions in the format (C, H, W),
    which is converted to a numpy array with (H, W, C) or (H, W) if C=1.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if len(tensor) == 6:
        return to_numpy(tensor[:3].permute(1, 2, 0)).squeeze()
    return to_numpy(tensor.permute(1, 2, 0)).squeeze()



class BcosModelBase:
    
    @classmethod
    def gradient_to_image(
        cls,
        image: "Tensor",
        linear_mapping: "Tensor",
        smooth: int = 0,
        alpha_percentile: float = 99.5,
    ) -> "np.ndarray":
        """
        From https://github.com/moboehle/B-cos/blob/0023500ce/interpretability/utils.py#L41.
        Computing color image from dynamic linear mapping of B-cos models.

        Parameters
        ----------
        image: Tensor
            Original input image (encoded with 6 color channels)
            Shape: [C, H, W] with C=6
        linear_mapping: Tensor
            Linear mapping W_{1\rightarrow l} of the B-cos model
            Shape: [C, H, W] same as image
        smooth: int
            Kernel size for smoothing the alpha values
        alpha_percentile: float
            Cut-off percentile for the alpha value

        Returns
        -------
        np.ndarray
            image explanation of the B-cos model.
            Shape: [H, W, C] (C=4 ie RGBA)
        """
        # shape of img and linmap is [C, H, W], summing over first dimension gives the contribution map per location
        contribs = (image * linear_mapping).sum(0, keepdim=True)
        contribs = contribs[0]
        # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
        rgb_grad = linear_mapping / (
            linear_mapping.abs().max(0, keepdim=True)[0] + 1e-12
        )
        # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
        rgb_grad = rgb_grad.clamp(0)
        # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
        rgb_grad = to_numpy(rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12))

        # Set alpha value to the strength (L2 norm) of each location's gradient
        alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
        # Only show positive contributions
        alpha = torch.where(contribs[None] < 0, torch.zeros_like(alpha) + 1e-12, alpha)
        if smooth:
            alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
        alpha = to_numpy(alpha)
        alpha = (alpha / np.percentile(alpha, alpha_percentile)).clip(0, 1)

        rgb_grad = np.concatenate([rgb_grad, alpha], axis=0)
        # Reshaping to [H, W, C]
        grad_image = rgb_grad.transpose((1, 2, 0))
        return grad_image

    @classmethod
    def plot_contribution_map(
        cls,
        contribution_map,
        ax=None,
        vrange=None,
        vmin=None,
        vmax=None,
        hide_ticks=True,
        cmap="bwr",
        percentile=100,
    ):
        """
        From https://github.com/moboehle/B-cos/blob/0023500cea7b/interpretability/utils.py#L78-L115
        For an example of how to use this, see docstring for `explain`.

        Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
        As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
        ranges from (-max(abs(contribution_map), max(abs(contribution_map)).

        Args:
            contribution_map: (H, W) matrix to visualise as contributions.
            ax: axis on which to plot. If None, a new figure is created.
            vrange: If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
                If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
                overwritten by vmin or vmax.
            vmin: Manually overwrite the minimum value for the colormap range instead of using -vrange.
            vmax: Manually overwrite the maximum value for the colormap range instead of using vrange.
            hide_ticks: Sets the axis ticks to []
            cmap: colormap to use for the contribution map plot.
            percentile: If percentile is given, this will be used as a cut-off for the attribution maps.
        Returns:
            The axis on which the contribution map was plotted.
        """
        assert (
            contribution_map.ndim == 2
        ), "Contribution map is supposed to only have 2 spatial dimensions."
        contribution_map = to_numpy(contribution_map)
        cutoff = np.percentile(np.abs(contribution_map), percentile)
        contribution_map = np.clip(contribution_map, -cutoff, cutoff)

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1)

        if vrange is None or vrange == "auto":
            vrange = np.max(np.abs(contribution_map.flatten()))
        im = ax.imshow(
            contribution_map,
            cmap=cmap,
            vmin=-vrange if vmin is None else vmin,
            vmax=vrange if vmax is None else vmax,
        )

        if hide_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        return ax, im

plot_contribution_map = BcosModelBase.plot_contribution_map
gradient_to_image = BcosModelBase.gradient_to_image

DEFAULTS = dict(percentile=99.95, smooth=1, cmap="bwr")


class Draw:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, canvas):
        self.draw(canvas, *self.args, **self.kwargs)

    @classmethod
    def draw(cls, *args, **kwargs):
        raise NotImplementedError("Need to define a drawing method.")


class DrawImg(Draw):
    @classmethod
    def draw(cls, canvas, img):
        canvas.imshow(to_numpy_img(img))


def set_defaults(dict, **defaults):
    for k, v in defaults.items():
        dict[k] = dict.get(k, v)


class DrawBoxes(Draw):
    @staticmethod
    def dilate(coords, dilation=0, img_size=224):
        scale = dilation / 2
        xmin, ymin, xmax, ymax = coords
        xdiff = xmax - xmin
        xmin = xmin - scale * xdiff
        xmax = xmax + scale * xdiff

        ydiff = ymax - ymin
        ymin = ymin - scale * ydiff
        ymax = ymax + scale * ydiff
        return max(0, xmin), max(0, ymin), min(img_size, xmax), min(img_size, ymax)

    @classmethod
    def draw(
        cls, canvas, bboxes, img_size=224, max_boxes=100, dilation=0, **rect_kwargs
    ):
        set_defaults(rect_kwargs, edgecolor="black", facecolor="none", lw=2)
        # Rescale to canvas size
        sw = (np.diff(canvas.extent[:2]) / img_size)[0]
        sh = (np.diff(canvas.extent[2:]) / img_size)[0]

        for coords in bboxes[:max_boxes]:
            xmin, ymin, xmax, ymax = cls.dilate(coords, dilation)

            x0 = xmin * sw + canvas.extent[0]
            y0 = canvas.extent[3] - ymin * sh

            w = (xmax - xmin) * sw
            h = (ymin - ymax) * sh

            box = patches.Rectangle((x0, y0), w, h, **rect_kwargs)
            canvas.ax.add_patch(box)


class DrawMask(Draw):
    @classmethod
    def draw(cls, canvas, mask, **mask_opts):
        set_defaults(mask_opts, color="green")
        color_convert = lambda x: plt_colors.to_rgb(x) if isinstance(x, str) else x

        cmap = LinearSegmentedColormap.from_list(
            f"mask_colors", [(1, 1, 1, 0), color_convert(mask_opts["color"])], N=2
        )
        canvas.imshow(to_numpy(mask), cmap=cmap)


class DrawContributionMap(Draw):
    @classmethod
    def draw(cls, canvas, model, img, tgt, **plotting_args):
        set_defaults(plotting_args, **DEFAULTS)
        contribs = model.explain(img, tgt)["contribution_map"]
        contribs = F.avg_pool2d(
            contribs,
            plotting_args["smooth"],
            padding=plotting_args["smooth"] // 2,
            stride=1,
        ).sum(list(range(len(contribs.shape) - 2)))
        plot_contribution_map(
            contribs,
            percentile=plotting_args["percentile"],
            ax=canvas,
            cmap=plotting_args["cmap"],
        )


class DrawColoredExp(Draw):
    @classmethod
    def draw(cls, canvas, model, img, tgt, **plotting_args):
        set_defaults(plotting_args, **DEFAULTS)
        canvas.imshow(
            to_numpy(
                model.explain(
                    img,
                    tgt,
                    smooth=plotting_args["smooth"],
                    alpha_percentile=plotting_args["percentile"],
                )["explanation"]
            )
        )


class DrawMeanCorrectedExp(Draw):
    @classmethod
    def draw(cls, canvas, model, img, tgt, **plotting_args):
        set_defaults(plotting_args, **DEFAULTS)
        with model.explanation_mode():
            im_var = Variable(img, requires_grad=True)
            out = model(im_var)
            (out - out.mean(1, keepdim=True))[0, tgt].backward()
            weights = im_var.grad
        canvas.imshow(
            to_numpy(
                gradient_to_image(
                    img[0],
                    weights[0],
                    smooth=plotting_args["smooth"],
                    alpha_percentile=plotting_args["percentile"],
                )
            )
        )
        im_var.grad = None
