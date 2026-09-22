# ==========================================================
# Experiment configuration
# ==========================================================
import sys
import os
import socket
import joblib

import datetime as dt
from pathlib import Path

from dataclasses import dataclass, asdict

import argparse
import textwrap

import numpy as np
import torch

WARMUP_EPOCHS = 0

WEIGHT_DECAY = 5.e-4
HREG_DECAY = 0.

LAMBDA_ANCHOR = 0.
LAMBDA_CENTER = 0.


@dataclass
class ExperimentConfig:
    """
    Experiment configuration including dataset, optimization
    and execution parameters.
    """

    bunch_file: str

    batch_size: int = 128
    epochs: int = 10
    reps: int = 1

    seed: int = 42
    
    gpu: str = "0"

    optimizer: str = "sgd"
    loss: str = "mse"
    resnet_model: str = '18'

    frac: float = 0.1
    lrate_factor: float = 1.0

    encoding: str = "ye"

    resampling_factor: float = 1.0
    
    save_results: bool = False
    
    weight_decay: float = WEIGHT_DECAY
    hreg_decay: float = HREG_DECAY
    
    init_noise: float = -1. 
    
    lambda_anchor: float = LAMBDA_ANCHOR
    lambda_center: float = LAMBDA_CENTER
    
    warmup_epochs: int = WARMUP_EPOCHS


def set_seed(seed: int):
    """
    Initialize NumPy and PyTorch random number generators.

    Ensures reproducible experiments across runs and repetitions.
    """

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        
# ==========================================================
# Argument parsing
# ==========================================================

def parse_args():
    """
    Parse command-line arguments for experiment execution.
    """

    parser = argparse.ArgumentParser(
        description="Train ResNet backbone on imbalanced datasets"
    )

    parser.add_argument(
        "-bf",
        "--bunch_file",
        required=True,
        type=str,
        help="joblib bunch file",
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=128,
        help="default: 128",
    )

    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="default: 10",
    )

    parser.add_argument(
        "-rp",
        "--reps",
        type=int,
        default=1,
        help="default: 1",
    )

    parser.add_argument(
        "-g",
        "--gpu",
        default="0",
        help="default: 0",
    )

    parser.add_argument(
        "-ops",
        "--optimizer_string",
        choices=["sgd", "adamw"],
        default="sgd",
        help="default: sgd",
    )

    parser.add_argument(
        "-l",
        "--loss",
        choices=["ce", "mse", "anchor_loss", "center_loss", "anchorcenter_loss", "hreg_loss", "anchorhreg_loss"],
        default="mse",
        help="default: mse; other losses: ce, anchor_loss, center_loss, anchorcenter_loss, hreg_loss, anchorhreg_loss"
    )

    parser.add_argument(
        "-rnm",
        "--resnet_model",
        type=str,
        choices=["18", "34", "50"],
        default="18",
        help="Pytorch resnet model to use; default: 18, use resnet18"
    )

    parser.add_argument(
        "-fr",
        "--frac",
        type=float,
        default=0.1,
        help="default: 0.1",
    )

    parser.add_argument(
        "-lrf",
        "--lrate_factor",
        type=float,
        default=1.0,
        help="default: 1.0",
    )

    parser.add_argument(
        "-wd",
        "--weight_decay",
        type=float,
        default=5.e-4,
        help="default: 5.e-4, ok for sgd on mnist; for adamw try higher values using lrate_factor",
    )

    parser.add_argument(
        "-hrd",
        "--hreg_decay",
        type=float,
        default=HREG_DECAY,
        help="default: 0.0; hreg decay doesn't apply",
    )

    parser.add_argument(
        "-la",
        "--lambda_anchor",
        type=float,
        default=LAMBDA_ANCHOR,
        help="weight of the Anchor penalty; default: 0.,  the penalty doesn't apply",
    )

    parser.add_argument(
        "-lc",
        "--lambda_center",
        type=float,
        default=LAMBDA_CENTER,
        help="weight of the Center penalty; default: 0., the penalty doesn't apply",
    )

    parser.add_argument(
        "-en",
        "--encoding",
        choices=["ohe", "ye"],
        default="ye",
        help="default: ye",
    )

    parser.add_argument(
        "-wu",
        "--warmup_epochs",
        type=int,
        default=0,
        help="numbre of warmup epochs; default: 0",
    )
    
    parser.add_argument(
        "-sv",
        "--save_results",
        type=int,
        default=0,
        help="default: 0, don't save",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed; default: 42",
    )
    
    parser.add_argument(
        "-rsf",
        "--resampling_factor",
        type=float,
        default=1.0,
        help=("Minority class resampling ratio; default: 1.0, no resampling and use standard Torch loader."
              "Doesn't seem to help getting good test accuracies."
              "Best not to use it"),
    )
    
    parser.add_argument(
        "-in",
        "--init_noise",
        type=str,
        default=-1.,
        help="Random noise to be added to optimal fc weights at initialization. Default: -1., use resnet default fc values",
    )
        
    return parser.parse_args()


# ==========================================================
# Configuration builder
# ==========================================================

def build_config(args):
    """
    Create an ExperimentConfig instance from parsed arguments.
    """
    return ExperimentConfig(
        bunch_file=args.bunch_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        reps=args.reps,
        gpu=args.gpu,
        optimizer=args.optimizer_string,
        loss=args.loss,
        resnet_model=args.resnet_model,
        frac=args.frac,
        lrate_factor=args.lrate_factor,
        weight_decay=args.weight_decay,
        hreg_decay=args.hreg_decay,
        lambda_anchor=args.lambda_anchor,
        lambda_center=args.lambda_center,
        warmup_epochs=args.warmup_epochs,
        encoding=args.encoding,
        save_results=bool(args.save_results),
        resampling_factor=args.resampling_factor,
        #frozen_weights=bool(args.frozen_weights)
        init_noise=args.init_noise
    )


# ==========================================================
# Paths
# ==========================================================

def get_paths(cfg):
    """
    Return dataset and results directories for the current host.

    Creates the results directory if it does not already exist.
    """
    host_name = socket.gethostname()

    print("\n.......... hostname:", host_name)

    if "spark" in host_name:

        data_dir = (
            "/home/jdorrons/data/"
            "tensorflow_datasets/"
        )

        results_root = (
            "/home/jdorrons/ongoing/"
            "nn_collapse/results/"
        )

    elif 'hoya' in host_name or 'casarrubuelos' in host_name:

        data_dir = (
            "/home/proyectos/ada2/"
            "jdorrons/data/"
            "tensorflow_datasets/"
        )

        results_root = (
            "/home/proyectos/ada2/"
            "jdorrons/ongoing/"
            "nn_collapse/"
        )

    else:
        sys.exit('unknown host')
        
    dataset_name = (
        "fashion_mnist"
        if "fashion_mnist" in cfg.bunch_file
        else "mnist"
    )

    results_dir = (
        f"{results_root}"
        #f"{dataset_name}_test/"
        f"{dataset_name}/"
        f"frac_{cfg.frac}_"
        f"bs_{cfg.batch_size}_"
        f"lrf_{cfg.lrate_factor}/"
        #f"la_{cfg.lambda_anchor}/"
    )

    Path(results_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    #folder to save results when only one rep 
    exp_1_dir = (
        f"/home/proyectos/ada2/jdorrons/ongoing/nn_collapse/nc-ye_exps/"
    )
    
    Path(exp_1_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("results_dir:", results_dir,
          "\n  exp_1_dir:", exp_1_dir)

    return data_dir, results_dir, exp_1_dir


# ==========================================================
# Device
# ==========================================================

def get_device(cfg):
    """
    Select the computation device specified by the configuration.

    Returns a CUDA device when available, otherwise CPU.
    """
    if torch.cuda.is_available():

        device = torch.device(
            f"cuda:{cfg.gpu}"
        )

        #print("Using GPU:", device)

        return device

    #print("Using CPU")

    return torch.device("cpu")


