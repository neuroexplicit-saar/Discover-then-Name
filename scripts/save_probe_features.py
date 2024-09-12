import os

import torch
from tqdm.auto import tqdm

from pprint import pprint
import os.path as osp

from dncbm.utils import common_init, get_img_model, get_probe_dataset
from dncbm import arg_parser


class FetchFeatures:
    def __init__(self, args=None):
        self.model, self.preprocess = get_img_model(args)
        self.args = args

    def dump_idxs(self):
        """ split the train indices into train and train_val;  for training and validation on the probe dataset"""
        assert (self.args.probe_split == "train")
        train_dataset = get_probe_dataset(self.args.probe_dataset, self.args.probe_split, self.args.probe_dataset_root_dir, self.preprocess)
        randperm = torch.randperm(len(train_dataset))
        train_prop = 0.9
        train_num = int(train_prop * len(train_dataset))
        train_idxs = randperm[:train_num]
        train_val_idxs = randperm[train_num:]

        assert (train_idxs.shape[0]+train_val_idxs.shape[0] == randperm.shape[0] == len(train_dataset))

        torch.save(train_idxs, os.path.join(
            self.args.probe_split_idxs_dir['img'], "train_idxs.pth"))
        torch.save(train_val_idxs, os.path.join(
            self.args.probe_split_idxs_dir['img'], "train_val_idxs.pth"))

    def get_probe_out(self, loader):
        count = 0
        with torch.no_grad():
            labels = None
            out = None

            for (inputs, idx) in tqdm(loader, desc="Processing batches", unit="batch"):
                count += inputs.shape[0]
                inputs = inputs.to(self.args.device)

                if out is None:
                    out = self.model.encode_image(inputs).detach().cpu()
                    labels = idx
                else:
                    out = torch.vstack(
                        (out, self.model.encode_image(inputs).detach().cpu()))
                    labels = torch.hstack((labels, idx))

                print(f" total data points: {count}")
            assert (labels.shape[0] == out.shape[0])
        return out, labels

    def save_probe_features(self, probe_dataset):
        "save the train and train_val and val features and labels"

        if not osp.exists(os.path.join(args.probe_split_idxs_dir['img'], "train_idxs.pth")):
            print(f"\n Labels alreday exist! \n")
            self.dump_idxs()

        # train split
        train_idxs = torch.load(os.path.join(
            args.probe_split_idxs_dir['img'], "train_idxs.pth"))
        train_val_idxs = torch.load(os.path.join(
            args.probe_split_idxs_dir['img'], "train_val_idxs.pth"))

        train_dataset = get_probe_dataset(
            probe_dataset, 'train', args.probe_dataset_root_dir, self.preprocess)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False)
        output, labels = self.get_probe_out(train_loader)

        # split the train data
        train_output = output[train_idxs]
        train_val_output = output[train_val_idxs]
        train_labels = labels[train_idxs]
        train_val_labels = labels[train_val_idxs]

        os.makedirs(args.probe_data_dir_activations["img"], exist_ok=True)
        torch.save(train_output, osp.join(
            args.probe_data_dir_activations["img"], f"train.pth"))
        torch.save(train_val_output, osp.join(
            args.probe_data_dir_activations["img"], f"train_val.pth"))

        torch.save(train_labels, os.path.join(
            args.probe_split_idxs_dir['img'], "all_labels_train.pth"))
        torch.save(train_val_labels, os.path.join(
            args.probe_split_idxs_dir['img'], "all_labels_train.pth"))
        del output

        # val split
        val_dataset = get_probe_dataset(
            args.probe_dataset, 'val', args.probe_dataset_root_dir, self.preprocess)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False)
        output, labels = self.get_probe_out(val_loader)
        torch.save(output, osp.join(
            args.probe_data_dir_activations["img"], f"val.pth"))
        torch.save(labels, os.path.join(
            args.probe_split_idxs_dir['img'], "all_labels_val.pth"))


if __name__ == '__main__':
    # Run this file if you want to save CC3M activation on CLIP
    parser = arg_parser.get_common_parser()
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    common_init(args)
    pprint(vars(args))

    fetch_act = FetchFeatures(args)
    # to save probe_features
    fetch_act.save_probe_features(args.probe_dataset)
