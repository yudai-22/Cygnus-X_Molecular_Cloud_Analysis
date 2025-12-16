import argparse
import importlib
import itertools
import os
import shutil
from itertools import product as product
from math import sqrt as sqrt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from PIL import ImageFile

import train_model
# from model_04 import Conv3dAutoencoder
from training_sub import weights_init

sys.path.append("/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Binary_classification/training/models")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of CAE")
    # 学習データの読み込みとモデルの書き込み
    parser.add_argument("--training_path", metavar="DIR", help="training_path", 
                        default="../data/processed_data/split_data/training_data/concatenate_data/training_data_concatenate.npy")
    parser.add_argument("--validation_path", metavar="DIR", help="validation_path", 
                        default="../data/processed_data/split_data/validation_data/concatenate_data/validation_data_concatenate.npy")
    parser.add_argument("--training_labels_path", metavar="DIR", help="training_labels_path", 
                        default="../data/processed_data/split_data/training_data/concatenate_data/training_labels_concatenate.npy")
    parser.add_argument("--validation_labes_path", metavar="DIR", help="validation_labels_path", 
                        default="../data/processed_data/split_data/validation_data/concatenate_data/validation_labels_concatenate.npy")
    parser.add_argument("--savedir_path", metavar="DIR", help="savedir path", 
                        default="/home/cygnus/fujimoto/Cygnus-X_Molecular_Cloud_Analysis/Binary_classification/training/save_dir")
    
    # minibatch
    parser.add_argument("--num_epoch", type=int, default=1000, help="number of total epochs to run (default: 1000)")
    parser.add_argument("--train_mini_batch", default=16, type=int, help="mini-batch size (default: 32)")
    parser.add_argument("--val_mini_batch", default=16, type=int, help="Validation mini-batch size (default: 128)")
    
    # random seed
    parser.add_argument("--random_state", "-r", type=int, default=123)
    
    # 学習率
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    
    # 潜在変数
    parser.add_argument("--latent_num", type=int, default=100)
    
    # Augmentation
    parser.add_argument("--augment_horizontal", type=bool, default=True)
    parser.add_argument("--augment_vertical", type=bool, default=True)
    parser.add_argument("--augment_velocity_axis", type=bool, default=False)
    
    # option
    parser.add_argument("--wandb_project", type=str, default="Binary_classification_matching")
    #parser.add_argument("--wandb_name", type=str, default="demo1")
    parser.add_argument("--wandb_name", type=str)
    
    # モデルの選択
    parser.add_argument("--model_file", type=str, default="Binary_classification", help="Name of the model file to import (without .py)")
    return parser.parse_args()


def load_model(model_file):
    try:
        model_module = importlib.import_module(model_file)
        return model_module.Binary_classification
    except ImportError as e:
        raise ImportError(f"Failed to import module {model_file}. Make sure the file exists and is in the Python path.") from e


# Training of SSD
def main(args):
    """Train CAE.

    :Example command:
    >>> python /home/filament/fujimoto/Cygnus-X_CAE/github_dir/FUGIN_cloud/train_main.py \
        --training_validation_path /home/filament/fujimoto/Cygnus-X_CAE/data/cygnusX_layer120_merged_data_flip.npy \
        --savedir_path /home/filament/fujimoto/Cygnus-X_CAE/save_dir \
        --train_mini_batch 8 \
        --val_mini_batch 8 \
        --latent_num 1000 \
        --wandb_project FUGIN_cloud \
        --wandb_name Cygnus_CAE_120 \
    """
    torch.manual_seed(args.random_state)
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.savedir_path):
        print("REMOVE FILES...")
        shutil.rmtree(args.savedir_path)
    os.makedirs(args.savedir_path, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "fits_random_state": args.random_state,
            "train_mini_batch": args.train_mini_batch,
            "val_mini_batch": args.val_mini_batch,
            "latent_num": args.latent_num,
        },
    )


    Binary_classification = load_model(args.model_file)
    model = Binary_classification(latent=args.latent_num)
    model.apply(weights_init)
    model.to(device)
    wandb.watch(model, log_freq=100)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False
    )

    train_model_params = {
        "model": model,
        "criterion": nn.MSELoss(),
        "optimizer": optimizer,
        "num_epochs": args.num_epoch,
        "args": args,
        "device": device,
        "run": run,
        "augment_horizontal"   : args.augment_horizontal, 
        "augment_vertical"     : args.augment_vertical, 
        "augment_velocity_axis": args.augment_velocity_axis
    }

    train_model.train_model(**train_model_params)

    artifact = wandb.Artifact("training_log", type="dir")
    artifact.add_dir(args.savedir_path)
    run.log_artifact(artifact, aliases=["latest", "best"])

    run.alert(title="学習が終了しました", text="学習が終了しました")
    run.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
