import os
import os.path as osp
import torch
import matplotlib.pyplot as plt
import os.path as osp

from dncbm.utils import common_init
from dncbm import arg_parser, config
from dncbm.visualization.canvas import Canvas
from dncbm.visualization.vis_utils import latex_bf

parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default= 'final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

data_dir = osp.join(config.analysis_dir, 'global_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'data')
vis_dir  = osp.join(config.analysis_dir, 'global_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'vis')
 
class_images = torch.load(osp.join(data_dir, "class_images.pt"))
class_texts = torch.load(osp.join(data_dir, "class_texts.pt"))
class_names = torch.load(osp.join(data_dir, "class_names.pt"))

print("Done")
Canvas.DEFAULTS["fontsize"] = 30
Canvas.DEFAULTS["use_latex"] = True
Canvas.DEFAULTS["save_path"] = vis_dir

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

######################################################################################################
bpad = 25
spad = 25
imsize = 100
text_width = 1.5 * imsize

top = 0
left = 0

rows = 3 # how many rows of images to show for a class
subcols = 3 # how many columns of images to show for a class
cols = 3 # how many classes to show in a single plot

name_map = {"imagenet": "ImageNet", "cifar10": "CIFAR10", "cifar100": "CIFAR100", "places365": "Places365"}
num_samples_to_show = int(args.probe_nclasses//cols)


for fig_idx in range(num_samples_to_show):
    print(f"Plotting stage: {fig_idx+1}/{num_samples_to_show}")
    canvas = Canvas.make(width=cols*((subcols * (imsize + spad) - spad) + 2 * bpad + text_width) - bpad, height=rows*(imsize+spad)-spad, left=left, top=top, scale=55)

    major_cols = canvas.horizontal_split(cols, bpad)

    for idx, col in enumerate(major_cols):
        col.show_frame(color="black", lw=2, alpha=1)

    with canvas.zoom((left, canvas.extent[1], top, canvas.extent[-1])):
        for grid_idx, col in enumerate(major_cols):
            class_name = class_names[grid_idx + fig_idx*cols]
            col.set_title("Class: " + latex_bf(class_name), fontweight="normal")

            with col.select((col.extent[0]+text_width+bpad, *col.extent[1:])):
                subgrid = col.even_grid(rows, subcols, spad)
                for row_idx, row in enumerate(subgrid):
                    for col_idx, cell in enumerate(row):
                        cell.imshow(class_images[grid_idx + fig_idx*cols][row_idx * subcols + col_idx].clip(0, 1))
            
            with col.select((col.extent[0], col.extent[0]+text_width+bpad, *col.extent[2:])):
                if not grid_idx:
                    col.set_ylabel("Top Concepts")
                specific_texts = class_texts[grid_idx + fig_idx*cols][::-1]
                for shift, text in enumerate(specific_texts):
                    col.make_text(.5, .1*shift + .05, text, ha="center", fontweight="normal")

    os.makedirs(vis_dir, exist_ok=True)
    canvas.save(f"fig_{fig_idx}_class_{class_name}.pdf")