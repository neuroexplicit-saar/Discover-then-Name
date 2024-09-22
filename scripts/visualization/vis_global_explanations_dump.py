import os
import os.path as osp
import torch
import torchvision
import clip

from dncbm import arg_parser, config, method_utils
from dncbm.visualization.drawing_functions import to_numpy_img
from dncbm.utils import get_printable_class_name, common_init, get_probe_dataset

parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default= 'final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

embeddings_path = osp.join(args.vocab_dir,f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth" )
vocab_txt_path  = osp.join(args.vocab_dir, "clipdissect_20k.txt")

_, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
un_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

topk=10 # how many concepts to show per class
grid_dim = 3
num_samples = grid_dim**2

method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)
all_concepts = method_obj.get_concepts()
all_cocnepts = all_concepts.to(args.device)
all_labels = method_obj.get_labels()
all_labels.to(args.device)

classifier_weights = method_obj.get_classifier_weights()
classifier_weights = classifier_weights.to(args.device)
mean_concept_strengths = all_concepts.mean(dim=0, keepdim=True)
all_concepts -= mean_concept_strengths

data_dir = osp.join(config.analysis_dir, 'global_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'data')

class_images = {idx: [] for idx in range(args.probe_nclasses)}
class_texts = {idx: [] for idx in range(args.probe_nclasses)}
class_names = {idx: "" for idx in range(args.probe_nclasses)}

for plt_idx, class_idx in enumerate(range(args.probe_nclasses)):
    class_name = get_printable_class_name(args.probe_dataset, class_idx)
    class_names[class_idx] = class_name

    this_class_concepts = all_concepts[all_labels == class_idx]
    top_strengths, top_indices = torch.topk(this_class_concepts.mean(dim=0), k=topk)

    for concept_pos_idx, concept_idx in enumerate(top_indices):
        class_texts[class_idx].append(
            method_obj.get_concept_name(concept_idx)[0])

    imgs_this_class = torch.where(all_labels == class_idx)[0]
    random_perm = torch.randperm(len(imgs_this_class))
    selected_images = imgs_this_class[random_perm[:num_samples]]
    for which_samples_id, img_idx in enumerate(selected_images):
        img = dataset[img_idx][0].unsqueeze(0)
        img = un_normalize(img)[0]
        class_images[class_idx].append(to_numpy_img(img))
    print(f"dumping data stage: {class_idx+1}/{args.probe_nclasses}")

os.makedirs(data_dir, exist_ok=True)
torch.save(class_images, osp.join(data_dir, "class_images.pt"))
torch.save(class_texts, osp.join(data_dir, "class_texts.pt"))
torch.save(class_names, osp.join(data_dir, "class_names.pt"))