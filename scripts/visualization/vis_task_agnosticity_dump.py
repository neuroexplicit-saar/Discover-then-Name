import time
import torchvision
import torch 
import numpy as np
import os.path as osp
import clip
import os

from dncbm import arg_parser, method_utils, config
from dncbm.utils import common_init, get_probe_dataset


start_time = time.time()
parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args, disable_make_dirs=True)

embeddings_path = osp.join(args.vocab_dir,f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth" )
vocab_txt_path  = osp.join(args.vocab_dir, "clipdissect_20k.txt")

topk=500
dataset_list = ['cifar10', 'cifar100','imagenet', 'places365']
_, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
un_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))
img_ids = {dataset: [] for dataset in dataset_list}

data_dir = osp.join(config.analysis_dir,'task_agnosticity', args.img_enc_name_for_saving, 'data')


def dump_for_our_method():
    print(f"Dumping concept name, vocab id, similarity value for each node for our method!")
    method_obj._compute_all_concept_embeddings_texts_and_indices()

    concept_names = method_obj.all_vocab_txt_selected[0]
    vocab_ids = method_obj.all_selected_indices[0]
    sim_vals = method_obj.all_name_similarities[0]
            
    os.makedirs(data_dir, exist_ok=True)
    torch.save(concept_names, osp.join(data_dir, 'concept_names.pt')) 
    torch.save(vocab_ids, osp.join(data_dir, 'vocab_ids.pt'))  
    torch.save(sim_vals, osp.join(data_dir, 'sim_vals.pt')) 
    return sim_vals, vocab_ids, concept_names

for dataset_idx, dataset_name in enumerate(dataset_list):
    print("/n")
    print(f"dataset name:{dataset_name}; dataset_id:{dataset_idx}/{len(dataset_list)}")
    
    args.probe_dataset = dataset_name
    common_init(args, disable_make_dirs=True)
    
    dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
    method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)

    sim_vals, vocab_ids, concept_names = dump_for_our_method()

    concept_strengths = method_obj.get_concepts()
    
    _, top_img_indices = torch.topk(concept_strengths, k=topk, dim=0)

    chosen_concept_strengths = torch.gather(concept_strengths, 0, top_img_indices)
    zero_indices = torch.where(chosen_concept_strengths== 0)
    row_zero_indices = zero_indices[0]
    col_zero_indices = zero_indices[1]
    top_img_indices[row_zero_indices, col_zero_indices]=-1
    top_img_indices_tmp = top_img_indices.T.reshape(-1).squeeze().tolist()

    img_ids[dataset_name] = top_img_indices_tmp
    assert (len(img_ids[dataset_name]) ==topk*concept_strengths.shape[1])

fname = f"top{topk}_img_ids_dict_{args.probe_split}.pt"

os.makedirs(data_dir, exist_ok=True)
whole_path_name = osp.join(data_dir, fname)
torch.save(img_ids, whole_path_name)
print(f"Saving img_ids_dict at {whole_path_name}")
print(f"Time taken: {np.round(time.time()-start_time,3)}")
