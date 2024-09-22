import matplotlib.pyplot as plt
import os.path as osp
import torch
import os

from dncbm.utils import common_init
from dncbm import arg_parser, config
from dncbm.visualization.canvas import Canvas


parser = arg_parser.get_common_parser()
args = parser.parse_args()

n_clusters_dict = {'imagenet': 1300, 'cifar10': 20, 'places365': 500}
args.n_clusters = n_clusters_dict[args.probe_dataset]
common_init(args, disable_make_dirs=True)

data_dir = osp.join(config.analysis_dir, 'meta_clusters', args.img_enc_name_for_saving, f"{args.probe_dataset}_n_clusters{args.n_clusters}", 'data')
vis_dir  = osp.join(config.analysis_dir, 'meta_clusters', args.img_enc_name_for_saving, f"{args.probe_dataset}_n_clusters{args.n_clusters}", 'vis')
os.makedirs(vis_dir, exist_ok=True)

cluster_images = torch.load(osp.join(data_dir, "cluster_images.pt"))
cluster_texts = torch.load(osp.join(data_dir, "cluster_texts.pt"))

keys_list = list(cluster_images.keys())
clusters_to_vis = min(200, args.n_clusters)

Canvas.DEFAULTS["use_latex"] = True
Canvas.DEFAULTS["save_path"] = vis_dir

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

bpad = 25
spad = 5
imsize = 100
text_width = 1.5 * imsize

top = 0
left = 0

rows = 3
subcols = 3
cols = 3

name_map = {"imagenet": "ImageNet", "cifar10": "CIFAR10", "cifar100": "CIFAR100", "places365": "Places365"}

for fig_idx in range(clusters_to_vis//3):
    if 3*fig_idx + 3 > len(keys_list):
        break
    canvas = Canvas.make(width=cols*((subcols * (imsize + spad) - spad) + 2 * bpad + text_width) - bpad,
                            height=rows*(imsize+spad)-spad, left=left, top=top, scale=55)

    major_cols = canvas.horizontal_split(cols, bpad)

    for idx, col in enumerate(major_cols):
        col.show_frame(color="black", lw=2, alpha=1)

    with canvas.zoom((left, canvas.extent[1], top, canvas.extent[-1])):
        for grid_idx, col in enumerate(major_cols):
            with col.select((col.extent[0]+text_width+bpad, *col.extent[1:])):
                col.set_title("Sampled Images from Cluster")
                subgrid = col.even_grid(rows, subcols, spad)

                for row_idx, row in enumerate(subgrid):
                    for col_idx, cell in enumerate(row):
                        cell.imshow(cluster_images[keys_list[grid_idx + 3 * fig_idx]][row_idx * subcols + col_idx].clip(0, 1))

            with col.select((col.extent[0], col.extent[0]+text_width+bpad, *col.extent[2:])):
                if not grid_idx:
                    col.set_ylabel("Top Concepts")
                    specific_texts = cluster_texts[keys_list[grid_idx + 3 * fig_idx]][::-1]
                for shift, text in enumerate(specific_texts):
                    col.make_text(.5, .1*shift + .25, text, ha="center", fontweight="normal")

    canvas.save(f"clusters_{fig_idx}.pdf")
    plt.close()
    