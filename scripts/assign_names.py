from dncbm import method_utils
from dncbm import arg_parser
from dncbm import utils
from dncbm import config
import os

parser = arg_parser.get_common_parser()

args = parser.parse_args()
utils.common_init(args)

embeddings_path = os.path.join(
    args.vocab_dir, f"embeddings_{args.img_enc_name_for_saving}_clipdissect_20k.pth")
vocab_txt_path = os.path.join(args.vocab_dir, f"clipdissect_20k.txt")

method_obj = method_utils.get_method(
    "ours", args, embeddings_path=embeddings_path, vocab_txt_path=vocab_txt_path, use_fixed_sae=True)
concept_name_similarity_matrix = method_obj.get_concept_name_similarity_matrix()[
    0]
all_concept_names = method_obj.vocab_txt_all[0]
top_concept_idxs = concept_name_similarity_matrix.argmax(axis=0)

with open(os.path.join(args.save_dir["img"], f"concept_names.csv"), "w") as f:
    for idx in range(top_concept_idxs.shape[0]):
        name = all_concept_names[top_concept_idxs[idx]]
        print(f"{idx},{name}")
        f.write(f"{idx},{name}\n")
