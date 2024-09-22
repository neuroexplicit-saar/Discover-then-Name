# import sys
import torchvision
import os.path as osp
import torch
import random
import numpy as np
from dncbm import config
from pathlib import Path
from dncbm.data_utils import probe_classnames
import os
import clip

import torch.utils


def save_activation_hook(model, input, output):
    """
    Hook to save intermediate activations
    """
    model.activations = output


def get_img_model(args):
    if args.img_enc_name.startswith('clip'):
        model, preprocess = clip.load(
            args.img_enc_name[5:], device=args.device)
    elif args.img_enc_name.startswith("resnet50"):
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
        preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(
            256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), normalize])
    return model, preprocess


def get_sae_ckpt(args, autoencoder):
    """
    Loads the SAE checkpoint given configuration in args
    """
    save_dir_ckpt = args.save_dir_sae_ckpts[args.modality]
    ckpt_path = osp.join(save_dir_ckpt, f'sparse_autoencoder_final.pt')
    print(f"Loading SAE checkpoint from: {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    autoencoder.load_state_dict(state_dict)
    return autoencoder


def get_probe_classifier_ckpt(args, which_ckpt=None, name_only=False):
    """
    Loads and returns the probe classifier checkpoint and filename, given args
    """
    if which_ckpt is None:
        which_ckpt = args.probe_classifier_which_ckpt

    checkpoint_save_path = osp.join(
        args.probe_cs_save_dir, args.probe_config_name, "on_concepts_ckpts")

    whole_ckpt_fname = osp.join(
        checkpoint_save_path, f"on_concepts_{which_ckpt}_{args.probe_config_name}.pt")
    if not name_only:
        print(f"Loading classifier checkpoint from: {checkpoint_save_path}")
        state_dict = torch.load(whole_ckpt_fname)

        return state_dict, whole_ckpt_fname
    else:
        return whole_ckpt_fname


def set_seed(seed):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_printable_class_name(probe_dataset, class_idx):
    """
    Returns cleaned up class names for visualizations
    """
    if probe_dataset == "places365":
        class_name = " ".join(
            probe_classnames.probe_classes_dict[probe_dataset][class_idx].split("/")[2:])
        class_name = " ".join(class_name.split("_")).capitalize()
    elif probe_dataset == "imagenet":
        class_name = probe_classnames.imagenet_classes_clip[class_idx]
        class_name = class_name.capitalize()
    else:
        class_name = probe_classnames.probe_classes_dict[probe_dataset][class_idx]
        class_name = class_name.capitalize()
    return class_name


def common_init(args, disable_make_dirs=False):
    """
    Performs initializations of variables common to several scripts, and creates directories where applicable
    """

    set_seed(args.seed)

    args.config_name = f"lr{args.lr}_l1coeff{args.l1_coeff}_ef{args.expansion_factor}_rf{args.resample_freq}_hook{args.hook_points[0]}_bs{args.train_sae_bs}_epo{args.num_epochs}"
    # CSV-compatible config name
    args.config_name_csv = f"{args.img_enc_name},{args.hook_points[0]},{args.sae_dataset},{args.lr},{args.l1_coeff},{args.expansion_factor},{args.resample_freq},{args.train_sae_bs},{args.num_epochs}"

    args.img_enc_name_for_saving = args.img_enc_name.replace('/', '')

    # Directory names
    args.autoencoder_input_dim_dict = config.autoencoder_input_dim_dict
    args.data_dir_root = config.data_dir_root
    args.save_dir_root = config.save_dir_root
    args.probe_cs_save_dir_root = config.probe_cs_save_dir_root
    args.vocab_dir  = config.vocab_dir
    args.analysis_dir = config.analysis_dir

    args.data_dir_activations = {}
    args.data_dir_activations["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0])

    args.probe_data_dir_activations = {}
    args.probe_data_dir_activations["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset, args.img_enc_name_for_saving, args.hook_points[0])

    args.probe_split_idxs_dir = {}
    args.probe_split_idxs_dir["img"] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset)
    
    args.ae_input_dim_dict_key = {}
    args.ae_input_dim_dict_key["img"] = f"{args.img_enc_name_for_saving}_{args.hook_points[0]}"

    args.save_dir = {}
    args.save_dir_sae_ckpts = {}

    args.save_dir["img"] = Path(osp.join(
        args.save_dir_root, f"SAEImg/{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}/{args.config_name}"))

    if not disable_make_dirs:
        os.makedirs(osp.join(args.save_dir_root,
                    f"SAEImg/{args.sae_dataset}/{args.img_enc_name_for_saving}/{args.hook_points[0]}"), exist_ok=True)
        # os.makedirs(osp.join(args.save_dir_root, f"SAEText/{args.sae_dataset}/{args.text_enc_name_for_saving}/{args.hook_points[0]}"), exist_ok=True)

    for modality in args.save_dir:
        if not disable_make_dirs:
            args.save_dir[modality].mkdir(exist_ok=True)
        args.save_dir_sae_ckpts[modality] = Path(
            osp.join(args.save_dir[modality], "sae_checkpoints"))

        if not disable_make_dirs:
            args.save_dir_sae_ckpts[modality].mkdir(exist_ok=True)

    args.enc_name = {}
    args.enc_name["img"] = args.img_enc_name
    args.enc_name_for_saving = {}
    args.enc_name_for_saving["img"] = args.img_enc_name_for_saving

    bias_str = "nobias"

    if args.probe_classification_loss == "CE" and args.probe_sparsity_loss is None:
        args.probe_config_name = f"lr{args.probe_lr}_bs{args.probe_train_bs}_epo{args.probe_epochs}_{bias_str}"
    else:
        args.probe_config_name = f"lr{args.probe_lr}_bs{args.probe_train_bs}_epo{args.probe_epochs}_{bias_str}_cl{args.probe_classification_loss}_sp{args.probe_sparsity_loss}_spl{args.probe_sparsity_loss_lambda}"

    args.probe_dataset_root_dir = config.probe_dataset_root_dir_dict[args.probe_dataset]

    args.probe_features_save_dir = osp.join(
        config.probe_cs_save_dir_root, args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0], "on_features", args.probe_dataset)

    args.probe_cs_save_dir = osp.join(
        config.probe_cs_save_dir_root, args.sae_dataset, args.img_enc_name_for_saving, args.hook_points[0], args.config_name, args.probe_dataset)

    args.probe_labels_dir = {}
    args.probe_labels_dir['img'] = osp.join(
        args.data_dir_root, 'activations_img', args.probe_dataset)

    args.probe_nclasses = config.probe_dataset_nclasses_dict[args.probe_dataset]

    args.probe_config_name_csv = f"{args.probe_lr},{args.probe_train_bs},{args.probe_epochs},{bias_str},{args.probe_classification_loss},{args.probe_sparsity_loss},{args.probe_sparsity_loss_lambda}"
    args.probe_csv_path = osp.join(config.probe_cs_save_dir_root, 'probe_results.csv')

def get_probe_dataset(probe_dataset, probe_split, probe_dataset_root_dir, preprocess_fn, split_idxs=None):
    """
    Loads and returns a downstream dataset given the dataset name, split, root directory, and preprocessing function
    """
    if probe_dataset == "imagenet":
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(probe_dataset_root_dir, probe_split), transform=preprocess_fn)
    elif probe_dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=os.path.join(probe_dataset_root_dir), train=probe_split == "train", download=False, transform=preprocess_fn)
    elif probe_dataset == "places365":
        if probe_split == 'train':
            suffix = '-standard'
        else:
            suffix = ''
        dataset = torchvision.datasets.Places365(
            root=os.path.join(probe_dataset_root_dir), split=f"{probe_split}{suffix}", download=False, small=True, transform=preprocess_fn)
    elif probe_dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=os.path.join(probe_dataset_root_dir), train=probe_split == "train", download=False, transform=preprocess_fn)
    else:
        raise NotImplementedError
    if split_idxs is not None:
        dataset = torch.utils.data.Subset(dataset, split_idxs)
    return dataset
