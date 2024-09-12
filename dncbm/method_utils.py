import torch
from dncbm.config import autoencoder_input_dim_dict
from sparse_autoencoder import SparseAutoencoder
import numpy as np
import os.path as osp
from dncbm.utils import common_init
import os


class MethodBase:

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        common_init(args, disable_make_dirs=True)

    def get_concepts(self):
        """ 
        Get all concept strengths in dataset split
        """
        raise NotImplementedError

    def get_logits(self):
        """ 
        Get all logits in dataset split
        """
        raise NotImplementedError

    def get_labels(self):
        """ 
        Get all labels in dataset split
        """
        if self.args.probe_split == "train":
            all_labels = torch.load(
                f"{self.args.probe_labels_dir['img']}/all_labels_train.pth")
            all_labels_train_val = torch.load(
                f"{self.args.probe_labels_dir['img']}/all_labels_train_val.pth")
            all_labels = torch.cat([all_labels, all_labels_train_val], dim=0)
        else:
            all_labels = torch.load(
                f"{self.args.probe_labels_dir['img']}/all_labels_{self.args.probe_split}.pth")
        return all_labels

    def get_concept_name(self, concept_idx):
        """ 
        Get concept text name given a concept ID
        """
        raise NotImplementedError

    def get_output_save_dir(self):
        """ 
        Get directory to save everything
        """
        raise NotImplementedError

    def get_classifier_weights(self):
        """ 
        Get linear classifier weights
        """
        raise NotImplementedError

    def get_concept_text_embedding(self, concept_idx):
        """ 
        Get text embedding given a concept ID
        """
        raise NotImplementedError

    def get_top_concept_indices_for_class(self, class_idx, num_indices=5):
        """
        Get top concept indices for each class based on coverage
        """
        raise NotImplementedError

    def get_concepts_from_features(self, x):
        """
        Given feature x, get concept
        """
        raise NotImplementedError

    def get_similarities(self):
        raise NotImplementedError

    def get_name_similarity(self, concept_idx):
        raise NotImplementedError

    def _get_contribs(self, split):
        suffix = f"_{split}.pt" if split == "train" else ".pt"
        all_concepts_fname = os.path.join(
            self.save_dir, f"all_concepts{suffix}")
        print(f"Loading {split} concepts from: {all_concepts_fname}")
        concepts = torch.load(all_concepts_fname)
        probe_labels_dir = self.args.probe_labels_dir["img"]
        if split == "train":
            labels_train = torch.load(os.path.join(
                probe_labels_dir, f"all_labels_train.pth"))
            labels_train_val = torch.load(
                osp.join(probe_labels_dir, f"all_labels_train_val.pth"))
            labels = torch.cat([labels_train, labels_train_val], dim=0)
        else:
            labels = torch.load(
                os.path.join(probe_labels_dir, f"all_labels_{split}.pth"))
        assert (len(concepts) == len(labels))
        print(f"Loading {split} labels from: {probe_labels_dir}")
        cum_sum_indices = torch.bincount(labels.long())
        sorting_order = labels.argsort()
        concepts = concepts[sorting_order]
        concepts_classwise = torch.split(
            concepts, cum_sum_indices.tolist())
        contribs_classwise = []
        for cidx, concept_classwise in enumerate(concepts_classwise):
            assert (concept_classwise.shape[1] ==
                    self.linear_layer_weights.shape[1])
            # For each class, multiply the concept strengths of images in that class with
            # weights of that class
            contribs_classwise.append(
                concept_classwise * self.linear_layer_weights[cidx].cpu())

        all_contribs = torch.stack(
            [c.clamp(min=0).sum(dim=0) for c in contribs_classwise])
        return all_contribs


class MethodOurs(MethodBase):

    def __init__(self, args, vocab_txt_path=None, embeddings_path=None, use_fixed_sae=False, use_sae_from_args=False, **kwargs):
        self.args = args
        assert (use_fixed_sae and (not use_sae_from_args)) or (
            (not use_fixed_sae) and use_sae_from_args)
        if use_fixed_sae:
            assert args.img_enc_name == "clip_RN50", "Fixed SAE only implemented for CLIP RN50, for other encoders, please pass the config via args"
            sae_config_to_use = "lr0.0005_l1coeff3e-05_ef8_rf10_hookout_bs4096_epo200"

            self.probe_config_dict = {'imagenet': "lr0.001_bs512_epo200_nobias_clCE_spL1_spl1.0",
                                      'cifar100': "lr0.01_bs512_epo200_nobias_clCE_spL1_spl1.0",
                                      'cifar10': "lr0.001_bs512_epo200_nobias_clCE_spL1_spl1.0",
                                      'places365': "lr0.001_bs512_epo200_nobias_clCE_spL1_spl1.0",
                                      'waterbirds100': 'lr0.1_bs512_epo200_nobias_clCE_spL1_spl10.0', }

            probe_config_to_use = self.probe_config_dict[self.args.probe_dataset]
            self._decode_config(sae_config_to_use, probe_config_to_use)
        else:
            sae_config_to_use = self.args.config_name
        super().__init__(args, **kwargs)

        print(f"SAE config_used: {sae_config_to_use}")

        self.sae_load_dir = osp.join(
            args.save_dir["img"], "..", sae_config_to_use)
        self.load_dir = osp.join(
            args.probe_cs_save_dir, "..", "..", sae_config_to_use, self.args.probe_dataset)

        state_dict_path = os.path.join(
            self.sae_load_dir, "sae_checkpoints", 'sparse_autoencoder_final.pt')
        self.state_dict = torch.load(state_dict_path, map_location=args.device)

        self.concept_layer = self._get_concept_layer()
        self.all_dic_vec = self.concept_layer.decoder.weight.detach().cpu().squeeze()
        self.decoder_bias = self.concept_layer.post_decoder_bias.bias.detach().cpu().squeeze()
        assert self.all_dic_vec.shape[0] == self.decoder_bias.shape[0]

        if embeddings_path is not None:
            assert vocab_txt_path is not None
            if type(embeddings_path) == str:
                self.all_embeddings = [torch.load(
                    embeddings_path, map_location='cpu')]
                self.vocab_txt_all = [np.genfromtxt(
                    vocab_txt_path, dtype=str, delimiter='\n')]
            elif type(embeddings_path) == list:
                self.all_embeddings = [torch.load(
                    e, map_location='cpu') for e in embeddings_path]
                self.vocab_txt_all = [np.genfromtxt(
                    t, dtype=str, delimiter='\n') for t in vocab_txt_path]
        else:
            print("Using default CLIP-Disssect embeddings and vocabulary")
            self.all_embeddings = [torch.load(os.path.join(
                args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth"))]
            self.vocab_txt_all = [np.genfromtxt(os.path.join(
                args.vocab_dir, f"clipdissect_20k.txt"), dtype=str, delimiter='\n')]

    def get_concepts(self, split=None):
        if split is None:
            split = self.args.probe_split
        if split == "train":
            concept_path1 = os.path.join(
                self.load_dir, "train", "all_concepts.pth")
            concept_path2 = os.path.join(
                self.load_dir, "train_val", "all_concepts.pth")
            train_concepts = torch.load(concept_path1)
            train_val_concepts = torch.load(concept_path2)
            img_concepts = torch.cat(
                [train_concepts, train_val_concepts], dim=0)
        else:
            concept_path = os.path.join(
                self.load_dir, split, "all_concepts.pth")
            img_concepts = torch.load(concept_path)
        img_concepts = img_concepts.squeeze(1)
        return img_concepts

    def get_logits(self):
        assert self.args.probe_split == "val"
        all_logits = torch.load(os.path.join(
            self.load_dir, self.args.probe_config_name, "stats", self.args.probe_split, "val_all_logits.pt"), map_location=self.args.device)
        return all_logits

    def get_concept_name_similarity_matrix(self):
        all_similarities = []
        for vocab_idx, vocab_specific_embedding in enumerate(self.all_embeddings):
            vocab_specific_embedding = vocab_specific_embedding.to(
                torch.float32)
            dic_vec = self.all_dic_vec  # (n_features, n_concepts)
            dic_vec /= dic_vec.norm(dim=0, keepdim=True)
            similarities = torch.matmul(
                vocab_specific_embedding, dic_vec)
            all_similarities.append(similarities.detach().cpu())
        return all_similarities

    def get_concept_name(self, concept_idx=None, dic_vec=None, return_sim=False, return_vocab_id=False):
        output = []
        sim = []
        vocab_id = []

        for vocab_idx, vocab_specific_embedding in enumerate(self.all_embeddings):
            vocab_specific_embedding = vocab_specific_embedding.to(
                torch.float32)
            if concept_idx is not None:
                dic_vec = self.all_dic_vec[:, concept_idx].unsqueeze(1)
            elif dic_vec is not None:
                dic_vec = dic_vec.unsqueeze(1)
            dic_vec /= dic_vec.norm(dim=0, keepdim=True)
            similarities = torch.matmul(
                vocab_specific_embedding, dic_vec).squeeze(dim=1)
            top_index = torch.argmax(similarities)
            output.append(self.vocab_txt_all[vocab_idx][top_index])
            sim.append(similarities[top_index])
            vocab_id.append(top_index)

        if return_sim:
            if return_vocab_id:
                return output, sim, vocab_id
            else:
                return output, sim
        else:
            return output

    def get_output_save_dir(self):
        return os.path.join(
            self.load_dir, self.probe_config_dict[self.args.probe_dataset])

    def get_classifier_weights(self):
        num_concepts = self.args.autoencoder_input_dim_dict[self.args.ae_input_dim_dict_key[self.args.modality]
                                                            ] * self.args.expansion_factor
        num_classes = self.args.probe_nclasses
        checkpoint_save_path = os.path.join(
            self.load_dir, self.args.probe_config_name, "on_concepts_ckpts")
        print(f"Loading classifier checkpoint from: {checkpoint_save_path}")
        state_dict = torch.load(os.path.join(checkpoint_save_path,
                                             f"on_concepts_{self.args.which_ckpt}_{self.args.probe_config_name}.pt"), map_location=self.args.device)
        classifier = torch.nn.Linear(
            num_concepts, num_classes, bias=False).to(self.args.device)
        classifier.load_state_dict(state_dict['model'])
        return classifier.weight

    def get_concept_text_embedding(self, concept_idx, use_dic_vec=False):
        if use_dic_vec:
            return self.all_dic_vec[:, concept_idx]
        output = []
        for vocab_idx in range(len(self.all_embeddings)):
            output.append(self.all_selected_embeddings[vocab_idx][concept_idx])
        return output

    def get_similarities(self):
        return self.name_similarities

    def get_name_similarity(self, concept_idx):
        output = []
        for vocab_idx in range(len(self.all_embeddings)):
            output.append(self.name_similarities[vocab_idx][concept_idx])
        return output

    def get_top_concept_indices_for_class(self, class_idx, num_indices=5):
        if not hasattr(self, 'top_indices_for_all_classes'):
            self.top_indices_for_all_classes = self.state_dict[
                "global_stats"][self.args.mod_type]["cov"]["node_idxs"][2]
        if num_indices > 10:
            raise ValueError(
                "Can currently only provide up to top 10 concept indices for a class")
        return self.top_indices_for_all_classes[class_idx][:num_indices]

    def get_concepts_from_features(self, x):
        concepts, _ = self.concept_layer.forward(x)
        concepts = concepts.squeeze(1)
        return concepts

    def _decode_config(self, sae_config, probe_config):
        if sae_config is not None:
            sae_config = sae_config.split("_")
            for item in sae_config:
                if item.startswith("lr"):
                    self.args.lr = float(item[2:])
                elif item.startswith("l1coeff"):
                    self.args.l1_coeff = float(item[7:])
                elif item.startswith("ef"):
                    self.args.expansion_factor = int(item[2:])
                elif item.startswith("rf"):
                    self.args.resample_freq = int(item[2:])
                elif item.startswith("hook"):
                    self.args.hook_points = [str(item[4:])]
                elif item.startswith("bs"):
                    self.args.train_sae_bs = int(item[2:])
                elif item.startswith("epo"):
                    self.args.num_epochs = int(item[3:])
                else:
                    raise ValueError(f"Invalid SAE config item: {item}")
        if probe_config is not None:
            probe_config = probe_config.split("_")
            for item in probe_config:
                if item.startswith("lr"):
                    self.args.probe_lr = float(item[2:])
                elif item.startswith("bs"):
                    self.args.probe_train_bs = int(item[2:])
                elif item.startswith("epo"):
                    self.args.probe_epochs = int(item[3:])
                elif item.startswith("nobias"):
                    self.args.probe_bias = False
                elif item.startswith("cl"):
                    self.args.probe_classification_loss = str(item[2:])
                elif item.startswith("spl"):
                    self.args.probe_sparsity_loss_lambda = float(item[3:])
                elif item.startswith("sp"):
                    self.args.probe_sparsity_loss = str(item[2:])
                else:
                    raise ValueError(f"Invalid probe config item: {item}")

    def _compute_all_concept_embeddings_texts_and_indices(self):
        multi_vocab_all_selected_embeddings = []
        multi_vocab_all_selected_indices = []
        multi_vocab_all_texts = []
        multi_vocab_all_name_similarities = []

        for vocab_idx, vocab_specific_embedding in enumerate(self.all_embeddings):
            all_selected_embeddings = []
            all_selected_indices = []
            all_texts = []
            all_name_similarities = []

            self.all_dic_vec /= self.all_dic_vec.norm(dim=0, keepdim=True)
            similarities = torch.matmul(
                vocab_specific_embedding, self.all_dic_vec).squeeze(dim=1)
            all_selected_indices = torch.argmax(similarities, dim=0).squeeze()
            all_selected_embeddings = vocab_specific_embedding[all_selected_indices]
            all_texts = self.vocab_txt_all[vocab_idx][all_selected_indices]
            all_name_similarities = similarities[all_selected_indices, np.arange(
                similarities.shape[1])]

            multi_vocab_all_selected_embeddings.append(
                all_selected_embeddings)
            multi_vocab_all_selected_indices.append(
                all_selected_indices)
            multi_vocab_all_texts.append(all_texts)
            multi_vocab_all_name_similarities.append(
                all_name_similarities)
        self.all_selected_embeddings, self.all_selected_indices, self.all_vocab_txt_selected, self.all_name_similarities = multi_vocab_all_selected_embeddings, multi_vocab_all_selected_indices, multi_vocab_all_texts, multi_vocab_all_name_similarities

    def _get_concept_layer(self):
        autoencoder_input_dim = autoencoder_input_dim_dict[self.args.ae_input_dim_dict_key["img"]]
        n_learned_features = int(
            autoencoder_input_dim * self.args.expansion_factor)
        autoencoder = SparseAutoencoder(n_input_features=autoencoder_input_dim,
                                        n_learned_features=n_learned_features, n_components=len(self.args.hook_points)).to(self.args.device)

        autoencoder.load_state_dict(self.state_dict)
        return autoencoder


def get_method(method, args, **kwargs):
    if method == "ours":
        return MethodOurs(args, **kwargs)
    else:
        raise ValueError(f"Invalid method: {method}")
