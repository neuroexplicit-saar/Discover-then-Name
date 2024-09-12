import argparse


def get_common_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, default="img", choices=["img"])
    parser.add_argument("--l1_coeff", type=float, default=3e-4)

    # Adam parameters (set to the default ones here)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam_beta_1", type=float, default=0.9)
    parser.add_argument("--adam_beta_2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--adam_weight_decay", type=float, default=0.0)

    parser.add_argument("--sae_dataset", type=str, default="cc3m")

    parser.add_argument("--img_enc_name", type=str, default='clip_RN50',
                        help='Name of the clip image encoder', choices=["clip_RN50", "clip_ViT-B/16", "clip_ViT-L/14"])
    parser.add_argument('--hook_points', nargs='*',
                        help='Name of the model hook points to get the activations from', default=['out'])

    parser.add_argument("--resample_freq", type=int,
                        default=500_000)  # 122_880_000
    parser.add_argument("--resample_dataset_size", type=int, default=819_200)
    parser.add_argument("--val_freq", type=int, default=50_000,help='number of samples after which to run validation')
    parser.add_argument("--ckpt_freq", type=int, default=500_000,
                        help='number of samples after which to save the checkpoint')

    # SAE related
    parser.add_argument("--train_sae_bs", type=int, default=4096,
                        help="batch size to train SAE")
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs to train the SAE")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--save_suffix", type=str, default="")

    parser.add_argument("--probe_dataset", type=str, default="imagenet")
    parser.add_argument("--vocab_type", type=str, default=None,
                        help="type of vocabulary to use for the probe dataset, if None uses the vocab depending on the method and the probe dataset")
    parser.add_argument("--method_name", type=str, default="ours",
                        choices=["ours", 'lfcbm', 'dclip', 'cdm', 'labo'])

    parser.add_argument("--probe_save_dir", type=str,
                        default="")
    parser.add_argument("--probe_lr", type=float, default=1e-4)
    parser.add_argument("--probe_train_bs", type=int, default=512)
    parser.add_argument("--probe_epochs", type=int, default=100,
                        help="number of epochs to train the linear probe")
    parser.add_argument("--probe_nclasses", type=int, default=1000,
                        help="number of classes in the probe dataset")
    parser.add_argument("--probe_on_features",
                        action="store_true", default=False, help="train probe on features you get from the feature extractor")
    parser.add_argument("--probe_split", type=str, default="train",
                        help="which split of the probe dataset to use for training or for analysis depending on the context")
    parser.add_argument("--probe_classification_loss",
                        type=str, default="CE", choices=["CE", "BCE"])
    parser.add_argument("--probe_sparsity_loss", type=str,
                        default=None, choices=["L1"])
    parser.add_argument("--probe_sparsity_loss_lambda", type=float, default=0)
    parser.add_argument("--probe_val_freq", type=int, default=1)
    parser.add_argument("--probe_eval_coverage_freq", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str, default="dncbm")

    return parser
