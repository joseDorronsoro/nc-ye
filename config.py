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

    frac: float = 0.1
    lrate_factor: float = 1.0

    encoding: str = "ye"

    save_results: bool = False


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
        default="sgd",
        help="default: sgd",
    )

    parser.add_argument(
        "-l",
        "--loss",
        choices=["ce", "mse"],
        default="mse",
        help="default: mse",
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
        "-en",
        "--encoding",
        choices=["ohe", "ye"],
        default="ye",
        help="default: ye",
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
    #parser.add_argument(
    #    "-sv",
    #    "--save_results",
    #    action="store_true",
    #)

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
        frac=args.frac,
        lrate_factor=args.lrate_factor,
        encoding=args.encoding,
        save_results=bool(args.save_results)
        #save_results=args.save_results,
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

    else:

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

    dataset_name = (
        "fashion_mnist"
        if "fashion_mnist" in cfg.bunch_file
        else "mnist"
    )

    results_dir = (
        f"{results_root}"
        f"{dataset_name}_test/"
        f"frac_{cfg.frac}_"
        f"{cfg.batch_size}_"
        f"{cfg.lrate_factor}/"
    )

    Path(results_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    print("results_dir:", results_dir)

    return data_dir, results_dir


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

        print("Using GPU:", device)

        return device

    print("Using CPU")

    return torch.device("cpu")


