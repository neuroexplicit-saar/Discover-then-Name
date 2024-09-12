import os
import os.path as osp
import copy

import torch
from tqdm import tqdm
import torchmetrics
import torchmetrics.classification
from torch.utils.data import TensorDataset
import torch.utils.data

from sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult
import datetime
import statistics

import wandb


from dncbm import arg_parser
from dncbm.utils import common_init
import torchmetrics

import fcntl


class ImageWiseCoverageEnergyPercent(torchmetrics.Metric):
    """
    Metric to compute, per image, how many concepts are needed to reach energy_percent of the original logit value.
    Concepts are selected greedily based on contribution to logit.
    Two modes are provided: Contribution to absolute value of the logit, and contribution to the positive part of the logit.
    """

    def __init__(self, energy_percent, use_ground_truth=False, use_only_positive=True):
        super().__init__()
        self.energy_percent = energy_percent
        self.use_ground_truth = use_ground_truth
        self.use_only_positive = use_only_positive
        self.add_state("coverage", default=[])
        self.add_state("coverage_fraction", default=[])

    def update(self, model, inputs, preds, target):
        if self.use_ground_truth:
            target_idx = target
        else:
            target_idx = torch.argmax(preds)
        contribs = model.weight.data[target_idx] * inputs
        assert len(contribs.shape) == 1
        if self.use_only_positive:
            contribs = contribs.clamp(min=0)
        else:
            contribs = contribs.abs()
        contribs = contribs.sort(descending=True).values
        contribs_cumsum = contribs.cumsum(dim=0)
        sufficient_position = torch.where(
            contribs_cumsum >= self.energy_percent*contribs.sum())[0][0]
        self.coverage.append(sufficient_position.item())
        self.coverage_fraction.append(
            (sufficient_position/contribs.shape[0]).item())

    def compute(self):
        if len(self.coverage) == 0:
            return None
        return statistics.fmean(self.coverage)


def eval_model(model, loader, num_classes, classification_loss_name, sparsity_loss_lambda, device, eval_coverage=False):
    """
    Evaluates model accuracy and sparsity during and after training.
    """
    with torch.no_grad():
        model.eval()
        accuracy_top1 = torchmetrics.classification.MulticlassAccuracy(
            num_classes=num_classes, top_k=1, average="micro").to(device)
        if args.probe_nclasses >= 5:
            accuracy_top5 = torchmetrics.classification.MulticlassAccuracy(
                num_classes=num_classes, top_k=5, average="micro").to(device)
        coverage_energy_wise = {}
        coverage_energy_wise_energies = [0.9, 0.95, 0.99]
        for energy in coverage_energy_wise_energies:
            coverage_energy_wise[energy] = {}
            coverage_energy_wise[energy]["positive"] = ImageWiseCoverageEnergyPercent(
                energy_percent=energy, use_ground_truth=False, use_only_positive=True).to(device)
            coverage_energy_wise[energy]["absolute"] = ImageWiseCoverageEnergyPercent(
                energy_percent=energy, use_ground_truth=False, use_only_positive=False).to(device)
        total_loss = 0
        total_classification_loss = 0
        total_sparsity_loss = 0
        total_batches = 0
        if classification_loss_name == "CE":
            classification_loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
        sparsity_loss_fn = torch.nn.L1Loss()
        for batch_idx, (test_X, test_y) in enumerate(tqdm(loader)):
            test_X = test_X.to(model.weight.dtype).to(device)
            test_y = test_y.long().to(device)
            out = model(test_X)
            classification_loss = classification_loss_fn(out, test_y)
            sparsity_loss = sparsity_loss_fn(
                model.weight.flatten(), torch.zeros_like(model.weight.flatten()))
            total_sparsity_loss += sparsity_loss.item()
            loss = classification_loss + sparsity_loss_lambda*sparsity_loss
            total_classification_loss += classification_loss.item()
            total_loss += loss.item()
            accuracy_top1.update(out, test_y)
            if args.probe_nclasses >= 5:
                accuracy_top5.update(out, test_y)
            if eval_coverage:
                for img_idx in range(test_X.shape[0]):
                    img = test_X[img_idx]
                    target = test_y[img_idx]
                    for energy in coverage_energy_wise_energies:
                        coverage_energy_wise[energy]["positive"].update(
                            model, img, out[img_idx], target)
                        coverage_energy_wise[energy]["absolute"].update(
                            model, img, out[img_idx], target)
            total_batches += 1
        zero_count_dict = {}
        zero_count = torch.where(model.weight.data.flatten() == 0)[0].shape[0]
        print(f"Zero count: {zero_count}")
        zero_count_dict[0] = zero_count
        for tol in [1e-9, 1e-6, 1e-3, 1e-1]:
            zero_count_tol = torch.where(
                torch.abs(model.weight.data.flatten()) < tol)[0].shape[0]
            print(
                f"Zero count for tol {tol}: {zero_count_tol}, out of {model.weight.data.flatten().shape[0]}")
            zero_count_dict[tol] = zero_count_tol
        acc_top1 = accuracy_top1.compute()
        if args.probe_nclasses >= 5:
            acc_top5 = accuracy_top5.compute()
        else:
            acc_top5 = 0.0
        if eval_coverage:
            coverage_energy_wise_metric_vals = {}
            for energy in coverage_energy_wise_energies:
                coverage_energy_wise_metric_vals[energy] = {}
                coverage_energy_wise_metric_vals[energy]["positive"] = coverage_energy_wise[energy]["positive"].compute(
                )
                coverage_energy_wise_metric_vals[energy]["absolute"] = coverage_energy_wise[energy]["absolute"].compute(
                )
                print(
                    f"Coverage for energy {energy} positive: {coverage_energy_wise_metric_vals[energy]['positive']}")
                print(
                    f"Coverage for energy {energy} absolute: {coverage_energy_wise_metric_vals[energy]['absolute']}")
        avg_loss = total_loss/total_batches
        avg_class_loss = total_classification_loss/total_batches
        avg_sparse_loss = total_sparsity_loss/total_batches
        model.train()
    if eval_coverage:
        return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, coverage_energy_wise_metric_vals, zero_count_dict
    return avg_loss, avg_class_loss, avg_sparse_loss, acc_top1, acc_top5, None, zero_count_dict


def main(args):

    if args.probe_on_features:
        wandb_project_name_prefix = f"Probe_training_on_features"
        probe_dir = os.path.join(
            args.probe_features_save_dir, args.probe_config_name)
        checkpoint_save_path = osp.join(
            probe_dir, "on_features_ckpts")
    else:
        wandb_project_name_prefix = f"Probe_training_on_concepts"
        probe_dir = os.path.join(
            args.probe_cs_save_dir, args.probe_config_name)
        checkpoint_save_path = osp.join(
            probe_dir, "on_concepts_ckpts")

    if args.use_wandb:
        wandb_project_name = f"{wandb_project_name_prefix}_{args.sae_dataset}_{args.img_enc_name_for_saving}_{args.hook_points[0]}_{args.probe_dataset}_{datetime.datetime.now().strftime('%Y-%m-%d')}{args.save_suffix}"
        wandb_dir = os.path.join(probe_dir, ".cache/")
        os.makedirs(wandb_dir, exist_ok=True)

        wandb.init(
            project=wandb_project_name,
            entity=args.wandb_entity,
            name=args.config_name+args.probe_config_name,
            dir=wandb_dir,
            config=args,
        )

    os.makedirs(checkpoint_save_path, exist_ok=True)

    num_input_nodes = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]]
    num_concepts = args.autoencoder_input_dim_dict[args.ae_input_dim_dict_key[args.modality]
                                                   ] * args.expansion_factor
    num_classes = args.probe_nclasses

    if args.probe_on_features:
        train_data = torch.load(
            osp.join(args.probe_data_dir_activations["img"], "train"))
        train_val_data = torch.load(
            osp.join(args.probe_data_dir_activations["img"], "train_val"))
        test_data = torch.load(
            osp.join(args.probe_data_dir_activations["img"], "val"))
        print(
            f"Getting {args.probe_dataset} features from: {args.probe_features_save_dir}")
    else:
        train_data = torch.load(
            osp.join(args.probe_cs_save_dir, "train", "all_concepts.pth"))
        train_val_data = torch.load(
            osp.join(args.probe_cs_save_dir, "train_val", "all_concepts.pth"))
        test_data = torch.load(
            osp.join(args.probe_cs_save_dir, "val", "all_concepts.pth"))
        print(
            f"Getting {args.probe_dataset} concepts from: {args.probe_cs_save_dir}")

    train_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train.pth"))
    train_val_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_train_val.pth"))
    test_labels = torch.load(
        osp.join(args.probe_labels_dir["img"], "all_labels_val.pth"))

    train_dataset = TensorDataset(train_data, train_labels)
    train_val_dataset = TensorDataset(train_val_data, train_val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.probe_train_bs, shuffle=True)
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset, batch_size=args.probe_train_bs, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.probe_train_bs, shuffle=False)

    if args.probe_on_features:
        model = torch.nn.Linear(
            num_input_nodes, num_classes, bias=False)
    else:
        model = torch.nn.Linear(
            num_concepts, num_classes, bias=False)
    model = model.train().to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.probe_lr)

    if args.probe_classification_loss == "CE":
        classification_loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    sparsity_loss_fn = None
    if args.probe_sparsity_loss is not None:
        sparsity_loss_fn = torch.nn.L1Loss()

    for e in range(args.probe_epochs):
        total_loss = 0
        total_classification_loss = 0
        total_sparsity_loss = 0
        total_batches = 0
        for batch_idx, (train_X, train_y) in enumerate(tqdm(train_loader)):
            total_batches += 1
            model.zero_grad()
            train_X = train_X.to(args.device)
            train_y = train_y.long().to(args.device)
            out = model(train_X)
            classification_loss = classification_loss_fn(out, train_y)
            if sparsity_loss_fn is not None:
                sparsity_loss = sparsity_loss_fn(
                    model.weight.flatten(), torch.zeros_like(model.weight.flatten()))
                total_sparsity_loss += sparsity_loss.item()
            else:
                sparsity_loss = 0
            total_classification_loss += classification_loss.item()
            loss = classification_loss + args.probe_sparsity_loss_lambda*sparsity_loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss/total_batches
        avg_class_loss = total_classification_loss/total_batches
        avg_sparse_loss = total_sparsity_loss/total_batches
        print(f"Epoch: {e+1}, Training Loss: {avg_loss}, Classification Loss: {total_classification_loss/total_batches}, Sparsity Loss: {total_sparsity_loss/total_batches}")

        if args.use_wandb:
            location = MetricLocation.TRAIN
            name = 'statistics'
            postfixes = ['Total Loss', 'Classification Loss', 'Sparsity Loss']
            metric_vals = [avg_loss, total_classification_loss /
                           total_batches, total_sparsity_loss/total_batches]
            metrics = []
            for p, mv in zip(postfixes, metric_vals):
                child_metrics = MetricResult(
                    location=location,
                    name=name,
                    postfix=p,
                    component_wise_values=torch.tensor([mv])
                )
                metrics.extend([child_metrics])

            if wandb.run is not None:
                log = {}
                for metric_result in metrics:
                    log.update(metric_result.wandb_log)
                if (e+1) % args.probe_val_freq == 0:
                    commit = False
                else:
                    commit = True
                wandb.log(
                    log,
                    step=e+1,
                    commit=commit,
                )

        if (e+1) % args.probe_val_freq == 0:
            if (e+1) % args.probe_eval_coverage_freq == 0:
                val_loss, val_class_loss, val_sparse_loss, val_acc_top1, val_acc_top5, coverage_energy_wise, zero_count_dict = eval_model(
                    model, train_val_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)
            else:
                val_loss, val_class_loss, val_sparse_loss, val_acc_top1, val_acc_top5, coverage_energy_wise, zero_count_dict = eval_model(
                    model, train_val_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device)

            print(
                f"Validation Loss: {val_loss}, Validation Classification Loss: {val_class_loss}, Validation Sparsity Loss: {val_sparse_loss}, Validation Accuracy Top1: {val_acc_top1}, Validation Accuracy Top5: {val_acc_top5}")

            if args.use_wandb:
                metrics = []
                postfix_list = ['Total Loss', 'Classification Loss', 'Sparsity Loss',
                                'accuracy top1', 'accuracy top5']
                stat_list = [val_loss, val_class_loss,
                             val_sparse_loss, val_acc_top1, val_acc_top5]
                for stat, postfix in zip(stat_list, postfix_list):
                    child_metrics = MetricResult(
                        location=MetricLocation.VALIDATE,
                        name='statistics',
                        postfix=postfix,
                        component_wise_values=torch.tensor([stat]))
                    metrics.extend([child_metrics])
                for zc in zero_count_dict.keys():
                    child_metrics = MetricResult(
                        location=MetricLocation.VALIDATE,
                        name='zero_count',
                        postfix=f'{zc}',
                        component_wise_values=torch.tensor([zero_count_dict[zc]]))
                    metrics.extend([child_metrics])
                if (e+1) % args.probe_eval_coverage_freq == 0:
                    for energy in coverage_energy_wise.keys():
                        for postfix in ['positive', 'absolute']:
                            child_metrics = MetricResult(
                                location=MetricLocation.VALIDATE,
                                name='coverage',
                                postfix=f'energy_{energy}_{postfix}',
                                component_wise_values=torch.tensor(
                                    [coverage_energy_wise[energy][postfix]]))
                            metrics.extend([child_metrics])

                if wandb.run is not None:
                    log = {}
                    for id, metric_result in enumerate(metrics):
                        log.update(metric_result.wandb_log)
                    wandb.log(
                        log,
                        step=e+1,
                        commit=True,
                    )

    val_loss, val_class_loss, val_sparse_loss, val_acc_top1, val_acc_top5, coverage_energy_wise, zero_count_dict = eval_model(
        model, train_val_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)

    test_loss, test_class_loss, test_sparse_loss, test_acc_top1, test_acc_top5, test_coverage_energy_wise, _ = eval_model(
        model, test_loader, num_classes, args.probe_classification_loss, args.probe_sparsity_loss_lambda, args.device, eval_coverage=True)

    saving_dict_final = {"model": copy.deepcopy(model.state_dict()), "epoch": e+1, "val_acc_top1": val_acc_top1,
                         "val_acc_top5": val_acc_top5, "val_loss": val_loss, "val_class_loss": val_class_loss, "val_sparse_loss": val_sparse_loss, "test_acc_top1": test_acc_top1,
                         "test_acc_top5": test_acc_top5, "test_loss": test_loss, "test_class_loss": test_class_loss, "test_sparse_loss": test_sparse_loss, 'train_loss': avg_loss, "train_class_loss": avg_class_loss, "train_sparse_loss": avg_sparse_loss, "val_coverage_energy_wise": coverage_energy_wise, "test_coverage_energy_wise": test_coverage_energy_wise, "zero_count_dict": zero_count_dict}

    if args.probe_on_features:
        prefix = 'on_features'
    else:
        prefix = 'on_concepts'

    torch.save(saving_dict_final, osp.join(checkpoint_save_path,
                                           f"{prefix}_final_{args.probe_config_name}.pt"))

    with open(args.probe_csv_path, "a") as outfile:
        fcntl.flock(outfile, fcntl.LOCK_EX)
        if args.probe_on_features:
            out_str_prefix = f"{args.img_enc_name},{args.hook_points[0]},{args.sae_dataset},features,{args.probe_dataset}," + \
                args.probe_config_name_csv + f","
        else:
            out_str_prefix = args.config_name_csv + f",{args.probe_dataset}," + \
                args.probe_config_name_csv + f","
        out_str = out_str_prefix + \
            f"{saving_dict_final['epoch']},{saving_dict_final['train_loss']:.4f},{saving_dict_final['train_class_loss']:.4f},{saving_dict_final['train_sparse_loss']:.4f},{saving_dict_final['val_loss']:.4f},{saving_dict_final['val_class_loss']:.4f},{saving_dict_final['val_sparse_loss']:.4f},{saving_dict_final['val_acc_top1']:.4f},{saving_dict_final['val_acc_top5']:.4f},{saving_dict_final['test_loss']:.4f},{saving_dict_final['test_class_loss']:.4f},{saving_dict_final['test_sparse_loss']:.4f},{saving_dict_final['test_acc_top1']:.4f},{saving_dict_final['test_acc_top5']:.4f}"
        for k in saving_dict_final['zero_count_dict']:
            print(k)
            out_str += f",{saving_dict_final['zero_count_dict'][k]}"
        if saving_dict_final['val_coverage_energy_wise'] is not None:
            for k in saving_dict_final['val_coverage_energy_wise']:
                for typ in ['positive', 'absolute']:
                    print(k, typ)
                    out_str += f",{saving_dict_final['val_coverage_energy_wise'][k][typ]}"
        else:
            for k in range(3):
                for typ in range(2):
                    out_str += f","
        for k in saving_dict_final['test_coverage_energy_wise']:
            for typ in ['positive', 'absolute']:
                print(k, typ)
                out_str += f",{saving_dict_final['test_coverage_energy_wise'][k][typ]}"
        outfile.write(out_str)
        outfile.write("\n")
        outfile.flush()
        fcntl.flock(outfile, fcntl.LOCK_UN)


parser = arg_parser.get_common_parser()
args = parser.parse_args()
common_init(args)
main(args)
