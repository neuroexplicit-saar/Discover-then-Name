import os
import torch
from tqdm.auto import tqdm
from pprint import pprint
import os.path as osp

# From custom
from dncbm.utils import common_init, get_img_model
from dncbm.data_utils.cc3m import (CC3MImg, CustomDataCollatorImg)
from dncbm import arg_parser


class FetchFeatures:
    def __init__(self, args=None):
        self.model, self.preprocess = get_img_model(args)
        self.args = args

    def get_cc3m_loader(self, shard, clip_preprocess, batch_size):
        collator = CustomDataCollatorImg()
        cc3m_obj = CC3MImg()
        dtset = cc3m_obj.get_wds_dataset(shard, clip_preprocess, batch_size, collator=collator)
        loader = cc3m_obj.get_dataloader(dtset, batch_size=None, shuffle=False)

        return loader
    
    def get_cc3m_out(self, loader):
        count = 0
        with torch.no_grad():
            idxs = []
            out = None

            for (inputs, idx) in tqdm(loader, desc="Processing batches", unit="batch"):
                count += inputs.shape[0]
                idxs.extend(idx)
                inputs = inputs.to(self.args.device)

                if out is None:
                    out = self.model.encode_image(inputs).detach().cpu()
                else:
                    out = torch.vstack((out, self.model.encode_image(inputs).detach().cpu()))
                print(f" total data points: {count}")
        return out, idxs
    
    def save_cc3m_features(self):
        data_dir_tar = osp.join(args.data_dir_root, "CC3M_TAR")
        train_path = "training"
        val_path = 'validation'

        train_shard     = os.path.join(data_dir_tar, train_path, "{00000..00299}.tar")
        train_val_shard = os.path.join(data_dir_tar, train_path, "{00300..00331}.tar")
        val_shard       = os.path.join(data_dir_tar, val_path, '{00000..00001}.tar')

        shard_list = [train_shard, val_shard, train_val_shard]
        type_list = ['train', 'val', 'train_val']
        save_dir_activations = args.data_dir_activations[args.modality]
        os.makedirs(save_dir_activations, exist_ok=True)

        for split, shard in zip(type_list, shard_list):
            loader = self.get_cc3m_loader(shard, clip_preprocess=self.preprocess, batch_size=args.batch_size)
            out,_ = self.get_cc3m_out(loader=loader)
            torch.save(out, osp.join(save_dir_activations, split))
            print(f"Save the activations from CLIP model to {save_dir_activations}")


if __name__ == '__main__':
    parser = arg_parser.get_common_parser()
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    common_init(args)
    pprint(vars(args))

    fetch_act = FetchFeatures(args)
    fetch_act.save_cc3m_features()
  


    