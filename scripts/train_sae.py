from dncbm.custom_pipeline import Pipeline
import os
from pathlib import Path

import torch
import numpy as np
import math
import datetime

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    SparseAutoencoder,
)
import wandb
from time import time

from dncbm.arg_parser import get_common_parser
from dncbm.utils import common_init


parser = get_common_parser()
args = parser.parse_args()
common_init(args)
start_time = time()


embeddings_path = f"/BS/language-explanations/work/concept_naming/embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth"
args.vocab_specific_embedding = torch.load(embeddings_path).to(args.device)


autoencoder_input_dim: int = args.autoencoder_input_dim_dict[
    args.ae_input_dim_dict_key[args.modality]]
n_learned_features = int(autoencoder_input_dim * args.expansion_factor)
autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                n_learned_features=n_learned_features, n_components=len(args.hook_points)).to(args.device)
print(f"Autoencoder created at {time() - start_time} seconds")

print(
    f"------------Getting Image activations from directory: {args.data_dir_activations[args.modality]}")
print(f"------------Getting Image activations from model: {args.img_enc_name}")


# We use a loss reducer, which simply adds up the losses from the underlying loss functions.
loss = LossReducer(LearnedActivationsL1Loss(
    l1_coefficient=float(args.l1_coeff),), L2ReconstructionLoss(),)
print(f"Loss created at {time() - start_time} seconds")

optimizer = AdamWithReset(
    params=autoencoder.parameters(),
    named_parameters=autoencoder.named_parameters(),
    lr=float(args.lr),
    betas=(float(args.adam_beta_1),
           float(args.adam_beta_2)),
    eps=float(args.adam_epsilon),
    weight_decay=float(args.adam_weight_decay),
    has_components_dim=True,
)

print(f"Optimizer created at {time() - start_time} seconds")
actual_resample_interval = 1
activation_resampler = ActivationResampler(
    resample_interval=actual_resample_interval,
    n_activations_activity_collate=actual_resample_interval,
    max_n_resamples=math.inf,
    n_learned_features=n_learned_features, resample_epoch_freq=args.resample_freq,
    resample_dataset_size=args.resample_dataset_size,
)

print(f"Activation resampler created at {time() - start_time} seconds")


if args.use_wandb:
    print("wandb started!")

    wandb_project_name = f"SAEImg_{args.sae_dataset}_{args.img_enc_name_for_saving}_{args.hook_points[0]}_{datetime.datetime.now().strftime('%Y-%m-%d')}{args.save_suffix}"

    print(f"wandb started! {wandb_project_name}")

    wandb_dir = os.path.join(args.save_dir[args.modality], ".cache/")
    wandb_path = Path(wandb_dir)
    wandb_path.mkdir(exist_ok=True)
    wandb.init(
        project=wandb_project_name,
        dir=wandb_dir,
        name=args.config_name,
        config=args,
        entity="text_concept_explanations",)

    wandb.define_metric("custom_steps")
    wandb.define_metric("train/loss_instability_across_batches",
                        step_metric="custom_steps")

    print(f"Wandb initialized at {time() - start_time} seconds")


pipeline = Pipeline(
    activation_resampler=activation_resampler,
    autoencoder=autoencoder,
    checkpoint_directory=Path(
        f"{args.save_dir_sae_ckpts[args.modality]}{args.save_suffix}"),
    loss=loss,
    optimizer=optimizer,
    device=args.device,
    args=args,
)
print(f"Pipeline created at {time() - start_time} seconds")

fnames = os.listdir(args.data_dir_activations[args.modality])
print(f"Getting fnames from {args.data_dir_activations[args.modality]}")

train_fnames = []
train_val_fnames = []
for fname in fnames:
    if fname.startswith(f"train_val"):
        train_val_fnames.append(os.path.join(
            os.path.abspath(args.data_dir_activations[args.modality]), fname))
    elif fname.startswith(f"train"):
        train_fnames.append(os.path.join(
            os.path.abspath(args.data_dir_activations[args.modality]), fname))
if args.val_freq == 0:
    train_fnames = train_fnames + train_val_fnames
    train_val_fnames = None

print(f"Train and Train_val fnames created at {time() - start_time} seconds")

# It takes the train activations and inside split it into train_activations and train_val_activations
pipeline.run_pipeline(
    train_batch_size=int(args.train_sae_bs),
    checkpoint_frequency=int(args.ckpt_freq),
    val_frequency=int(args.val_freq),
    num_epochs=args.num_epochs,
    train_fnames=train_fnames,
    train_val_fnames=train_val_fnames,
    start_time=start_time,
    resample_epoch_freq=args.resample_freq,
)

print(f"-------total time taken------ {np.round(time()-start_time,3)}")
