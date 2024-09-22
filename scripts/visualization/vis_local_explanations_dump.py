import os.path as osp
import torch
import torchvision
import os
import clip

from dncbm import method_utils, arg_parser, config
from dncbm.visualization.drawing_functions import to_numpy_img, to_numpy
from dncbm.utils import get_printable_class_name, common_init, get_probe_dataset

parser = arg_parser.get_common_parser()
parser.add_argument("--which_ckpt", type=str, default='final')
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

topk = 5 # number of text explanations per image
num_samples = 1000 # total number of plots to be done

embeddings_path = osp.join(args.vocab_dir,f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth" )
vocab_txt_path = osp.join(args.vocab_dir, "clipdissect_20k.txt")

_, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
un_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)

all_concepts = method_obj.get_concepts()
all_concepts = all_concepts.to(args.device)
all_labels = method_obj.get_labels()
all_labels = all_labels.to(args.device)

classifier_weights = method_obj.get_classifier_weights()
classifier_weights = classifier_weights.to(args.device)
data_dir = osp.join(config.analysis_dir, 'local_explanations', args.img_enc_name_for_saving, args.probe_dataset, 'data')

local_images = {}
local_class_name = {}
local_concepts = {}
local_img_idxs = {}

randperm = torch.randperm(len(all_labels))[:num_samples]

for img_pos_idx, img_idx in enumerate(randperm):
    concept_strengths = all_concepts[img_idx]
    contribs = concept_strengths * classifier_weights
    pred_class = torch.argmax(contribs.sum(dim=1)).item()
    top_strengths, top_indices = torch.topk(contribs[pred_class], k=topk)
    concept_texts = []

    for concept_pos_idx, concept_idx in enumerate(top_indices):
        concept_text = method_obj.get_concept_name(concept_idx)[0]
        concept_texts.append(concept_text)

    local_img_idxs[img_pos_idx] = img_idx
    img = dataset[img_idx][0].unsqueeze(0)
    img = to_numpy_img(un_normalize(img)[0])
    local_images[img_pos_idx] = img
    class_name = get_printable_class_name(args.probe_dataset, pred_class)

    local_class_name[img_pos_idx] = class_name
    local_concepts[img_pos_idx] = {text: strength for text, strength in zip(concept_texts, to_numpy(top_strengths))}

os.makedirs(data_dir, exist_ok=True) 
torch.save(local_class_name, osp.join(data_dir, f'local_class_name_{args.probe_split}.pt'))
torch.save(local_images, osp.join(data_dir, f'local_images_{args.probe_split}.pt'))
torch.save(local_concepts, osp.join(data_dir, f'local_concepts_{args.probe_split}.pt'))
torch.save(local_img_idxs, osp.join(data_dir, f'local_img_idxs_{args.probe_split}.pt'))

        
