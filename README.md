<h1 align="center">Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery</h1>

<div align="center">
<a href="https://sukrutrao.github.io">Sukrut Rao*</a>,
<a href="https://swetamahajan.github.io">Sweta Mahajan*</a>,
<a href="https://moboehle.github.io">Moritz Böhle</a>,
<a href="https://people.mpi-inf.mpg.de/~schiele">Bernt Schiele</a>

<h3 align="center">
European Conference on Computer Vision (ECCV) 2024
</div>
</h3>
  
<h3 align="center">
<a href="https://arxiv.org/abs/2407.14499">Paper</a>
|
<a href="https://github.com/neuroexplicit-saar/discover-then-name">Code</a>
|
<a href="https://sukrutrao.github.io/publication/discover-then-name-task-agnostic-concept-bottlenecks-via-automated-concept-discovery/DiscoverThenName_Poster.pdf">Poster</a>
</h3>
</p>

## Setup

### Prerequisites

All the dependencies and packages can be installed using pip. The code was tested using Python 3.10.

### Installing the Packages

Use:

```bash
pip install -r requirements.txt
pip install -e sparse_autoencoder/
pip install -e .
```

### Dataset for training Sparse Autoencoder (CC3M)

#### Download the CC3M tar file to train the SAE
Note: Number of downloaded paired dataset might be less than we used for our training as we downloaded the dataset in December, 2023. And as of now some more urls might be invalid. 

1) Download the ‘Train_GCC-training.tsv’ and ‘Validation_GCC-1.1.0-Validation.tsv’ from  https://ai.google.com/research/ConceptualCaptions/download by clicking on training split and validation split. 

2) Change their names to cc3m_training.tsv and cc3m_validation.tsv 

3) For training dataset: 
    ```bash
    sed -i '1s/^/caption\turl\n/' cc3m_training.tsv 
    img2dataset --url_list cc3m_training.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder training --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
    ``` 

4) for validation dataset:
    ```bash
    sed -i '1s/^/caption\turl\n/' cc3m_validation.tsv 
    img2dataset --url_list cc3m_validation.tsv --input_format "tsv" --url_col "url" --caption_col "caption" --output_format webdataset --output_folder validation --processes_count 16 --thread_count 64 --image_size 256 --enable_wandb True
    ```


### Datasets for training downstream probes

These are the datasets on which linear probes are trained on the learnt concept bottleneck to form a concept bottleneck model (CBM). In our paper, we use four datasets: Places365, ImageNet, CIFAR10, CIFAR100. Instructions for running experiments on these datasets is provided below, for other datasets you may need to define your own utils.

* Download the respective datasets:
    * [Places365](https://pytorch.org/vision/main/generated/torchvision.datasets.Places365.html)
    * [ImageNet](https://www.image-net.org/)
    * [CIFAR10](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
    * [CIFAR100](https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html)
* Set the paths to the datasets in `config.py`.




## Usage

The following shows example usage with CLIP ResNet-50 as the model, CC3M as the dataset for training the SAE, and Places365 as the dataset for downstream classification.

### Training a Sparse Autoencoder (SAE)


#### Save the CLIP features on CC3M to train the SAE on 

```bash
python scripts/save_cc3m_features.py --img_enc_name clip_RN50 
```

#### Train the SAE

```bash
python scripts/train_sae_img.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --num_epochs 200 --resample_freq 10 --ckpt_freq 0 --val_freq 1 --train_sae_bs 4096
```

### Assigning Names to Concepts

```bash
python scripts/assign_names.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --num_epochs 200 --resample_freq 10 --train_sae_bs 4096
```
 
### Training a Linear Probe for the Concept Bottleneck Model

#### Save the CLIP features of probe dataset

```bash
python scripts/save_probe_features.py --img_enc_name clip_RN50  --probe_dataset places365
```

#### Save concept strengths using the trained SAE

```bash
python scripts/save_concept_strengths.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --num_epochs 200  --resample_freq 10  --train_sae_bs 4096  --probe_dataset places365 --probe_split train
```

#### Train the probe on the saved concept strengths

```bash
python scripts/train_linear_probe.py --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 8 --img_enc_name clip_RN50 --resample_freq 10 --train_sae_bs 4096 --num_epochs 200 --ckpt_freq 0 --val_freq 1 --probe_lr 1e-2  --probe_sparsity_loss_lambda 0.1 --probe_classification_loss 'CE' --probe_epochs 200 --probe_sparsity_loss L1 --probe_eval_coverage_freq 50 --probe_dataset places365
```


## Acknowledgements

This repository uses code from the following repositories:

* [openai/CLIP](https://github.com/openai/CLIP)
* [ai-safety-foundation/sparse_autoencoder](https://github.com/ai-safety-foundation/sparse_autoencoder/)

## Citation

Please cite as follows:

```tex
@inproceedings{Rao2024Discover,
    author    = {Rao, Sukrut and Mahajan, Sweta and B\"ohle, Moritz and Schiele, Bernt},
    title     = {Discover-then-Name: Task-Agnostic Concept Bottlenecks via Automated Concept Discovery},
    booktitle = {European Conference on Computer Vision},
    year      = {2024}
}
```