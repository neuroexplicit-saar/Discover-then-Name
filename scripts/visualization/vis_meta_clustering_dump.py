import os
import torch
import clip
import torchvision
from sklearn.cluster import KMeans
import os.path as osp

from dncbm import method_utils, config, arg_parser
from dncbm.utils import common_init, get_probe_dataset


class GetConceptClusters:
    def __init__(self, class_concepts, n_clusters):
        super().__init__()
        self.class_concepts = class_concepts
        self.n_clusters = n_clusters

    def get_clusters(self):
        class_concepts = self.class_concepts
        kmeans = KMeans(n_clusters=self.n_clusters, verbose=1)
        kmeans.fit(class_concepts.detach().cpu().numpy())
        return kmeans.labels_

parser = arg_parser.get_common_parser()
args = parser.parse_args()

n_clusters_dict = {'imagenet': 1300, 'cifar10': 20, 'places365': 500}
args.n_clusters = n_clusters_dict[args.probe_dataset]
common_init(args, disable_make_dirs=True)

embeddings_path = osp.join(args.vocab_dir,f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth" )
vocab_txt_path  = osp.join(args.vocab_dir, "clipdissect_20k.txt")

data_dir = osp.join(config.analysis_dir, 'meta_clusters', args.img_enc_name_for_saving, f"{args.probe_dataset}_n_clusters{args.n_clusters}", 'data')
os.makedirs(data_dir, exist_ok=True)

_, preprocess = clip.load(args.img_enc_name[5:], device=args.device)
dataset = get_probe_dataset(args.probe_dataset, args.probe_split, args.probe_dataset_root_dir, preprocess_fn=preprocess)
un_normalize = torchvision.transforms.Normalize((-0.48145466/0.26862954, -0.4578275/0.26130258, -0.40821073/0.27577711), (1/0.26862954, 1/0.26130258, 1/0.27577711))

method_obj = method_utils.get_method(args.method_name, args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)
concept_strengths = method_obj.get_concepts()  # (n_samples, n_cocnepts)

# making clusters
get_concept_clusters = GetConceptClusters(concept_strengths, args.n_clusters)
if not osp.exists(osp.join(data_dir, "cluster_labels.pt")):
    cluster_labels = get_concept_clusters.get_clusters() # same length as the number of samples in the dataset
    cluster_labels = torch.tensor(cluster_labels)
    torch.save(cluster_labels, osp.join(data_dir, "cluster_labels.pt"))
else:
    cluster_labels = torch.load(osp.join(data_dir, "cluster_labels.pt"))

concept_strengths -= concept_strengths.mean(dim=0)
clusters_to_vis = min(200, args.n_clusters)
num_imgs_per_cluster = 9
num_concepts_to_show_per_cluster = 6

assert (cluster_labels.shape[0] == len(dataset))
cluster_images = {}
cluster_texts = {}

for cluster_idx in range(clusters_to_vis):
    cluster_idxs = torch.where(cluster_labels == cluster_idx)[0]
    if len(cluster_idxs) >= num_imgs_per_cluster:
        cluster_images[cluster_idx] = []
        cluster_texts[cluster_idx] = []
        cluster_concept_strengths = concept_strengths[cluster_idxs]
        cluster_concept_strengths_mean = cluster_concept_strengths.mean(dim=0)
        topk_values, topk_idxs = torch.topk(cluster_concept_strengths_mean, k=num_concepts_to_show_per_cluster)

        for concept_id in topk_idxs:
            cluster_texts[cluster_idx].append(method_obj.get_concept_name(concept_id)[0])

        shuffle_indices = torch.randperm(len(cluster_idxs))
        cluster_idxs = cluster_idxs[shuffle_indices].tolist()

        for ax_pos_idx, img_pos_idx in enumerate(cluster_idxs[:num_imgs_per_cluster]):
            img = dataset[img_pos_idx][0].unsqueeze(0)
            img = un_normalize(img)[0].detach().cpu().permute(1, 2, 0).numpy()
            cluster_images[cluster_idx].append(img)

torch.save(cluster_images, osp.join(data_dir, "cluster_images.pt"))
torch.save(cluster_texts,  osp.join(data_dir, "cluster_texts.pt"))