import os  
import torch
import os
from torch.utils.data import TensorDataset
from dncbm import arg_parser
from sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm
import os.path as osp

from dncbm.utils import common_init, get_sae_ckpt


def save_concept_strengths(args, is_cc3m=False):
    if is_cc3m:
        features_path = os.path.join(args.data_dir_activations["img"], args.probe_split)
    else:
        features_path = os.path.join(args.probe_data_dir_activations["img"], args.probe_split)
    all_features = torch.load(features_path)

    print(f"Loaded concepts from: {features_path}")

    dataset = TensorDataset(all_features)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)

    autoencoder_input_dim = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
    autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim, n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)

    autoencoder = get_sae_ckpt(args, autoencoder)
    all_concepts = None

    with torch.no_grad():
        for features in tqdm(loader):
            features = features[0].to(args.device)
            concepts, reconstructions = autoencoder(features)
            concepts, reconstructions = concepts.squeeze(), reconstructions.squeeze()
           
            if all_concepts is None:
                all_concepts = concepts.detach().cpu()
            else:
                all_concepts = torch.vstack((all_concepts, concepts.detach().cpu()))

    whole_all_concepts_fname = os.path.join(args.probe_cs_save_dir, args.probe_split, "all_concepts.pth")
    os.makedirs(osp.dirname(whole_all_concepts_fname), exist_ok=True)
    torch.save(all_concepts, whole_all_concepts_fname)
    print(f"Saved concepts at: {whole_all_concepts_fname}")

if __name__ == "__main__":
    parser = arg_parser.get_common_parser()
    args = parser.parse_args()
    common_init(args)
    save_concept_strengths(args, is_cc3m= False)
