

autoencoder_input_dim_dict = {'clip_RN50_out': 1024,
                              'clip_ViT-B16_out': 512,  
                              'clip_ViT-L14_out': 768, }

data_dir_root = './data'
save_dir_root = './SAE'
probe_cs_save_dir_root = './probe'
vocab_dir = './vocab'
analysis_dir = './analysis'



probe_dataset_root_dir_dict = {
    "places365": "/path/to/dataset/",
    "imagenet": "/path/to/dataset/",
    "cifar10": "/path/to/dataset/",
    "cifar100": "/path/to/dataset/",
}

probe_dataset_nclasses_dict = {"places365": 365,
                               'imagenet': 1000, "cifar10": 10, "cifar100": 100, }
