import os
import matplotlib.pyplot as plt
import os.path as osp
import torch
import clip
import torchvision

from dncbm.utils import common_init, get_probe_dataset
from dncbm import arg_parser
from dncbm.utils import get_printable_class_name
from dncbm import config
from dncbm import method_utils
from dncbm.visualization.canvas import Canvas
from dncbm.visualization.vis_utils import latex_bf


parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default= 'final', help='needed to decide which checkpoint of the Sparse autoencoder to load')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

embeddings_path = osp.join(args.vocab_dir,f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth" )
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")

_, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
un_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))
method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)

all_labels = method_obj.get_labels()
all_labels = all_labels.to(args.device)

data_dir = osp.join(config.analysis_dir,'local_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'data')
vis_dir  = osp.join(config.analysis_dir,'local_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'vis')

local_class_name = torch.load(osp.join(data_dir,f'local_class_name_{args.probe_split}.pt'))
local_images     = torch.load(osp.join(data_dir,f'local_images_{args.probe_split}.pt'))
local_concepts   = torch.load(osp.join(data_dir,f'local_concepts_{args.probe_split}.pt'))
local_img_idxs   = torch.load(osp.join(data_dir,f'local_img_idxs_{args.probe_split}.pt'))

Canvas.DEFAULTS["use_latex"] = True
Canvas.DEFAULTS["save_path"] = vis_dir

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

pad = 25
imsize = 100
top = 0
left = 0
rows = 1
cols = 2 # one column to plot the image and another to list down the most contributing concepts
bpad = 25

for fig_idx, (img_idx, class_name) in enumerate(local_class_name.items()):

    gt_class_name = get_printable_class_name(args.probe_dataset, int(all_labels[local_img_idxs[img_idx]].item()))
    canvas = Canvas.make(width=cols*(imsize + pad) - pad, height=imsize, left=left, top=top, scale=55)

    col1, col2 = canvas.horizontal_split(cols, bpad)
    col1.set_title(f"Pred: {class_name}, GT: {gt_class_name}", y_off=.015, fontsize=40)
    col2.set_title(f"Most Strongly Contributing Concepts", y_off=.015, fontsize=40)
    col1.imshow(local_images[img_idx].clip(0, 1))
    ax = canvas.ax.inset_axes([col2.extent[0], col2.extent[2], col2.extent[1] - col2.extent[1-1], col2.extent[3]-col2.extent[3-1]], transform=canvas.ax.transData)

    cs = []
    ss = []
    for c, s in local_concepts[img_idx].items():
        cs.append(c)
        ss.append(s)
    ax.barh(range(len(cs))[::-1], ss, color=(.1, .5, .4, 1))  # , width= 0.4)
    
    ax.axis("off")
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.1)
    for ypos, text, strength in zip(range(len(cs))[::-1], cs, ss):
        ax.text(.025 * ax.get_xlim()[1], ypos, latex_bf(text), fontsize=40, color="white", va="center") 
        ax.text(.025 * ax.get_xlim()[1] + strength, ypos, latex_bf(f"{strength:+.2f}"), fontsize=30, color="black", va="center")

    os.makedirs(vis_dir, exist_ok=True)
    canvas.save(f"local_{fig_idx}_pred_{class_name}_GT_{gt_class_name}.pdf")
    plt.close()

