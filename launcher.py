#!/usr/bin/env python
# coding: utf-8
########################################################################################################
import sys
import os

import numpy as np
import joblib

import datetime as dt
from pathlib import Path

from dataclasses import asdict

from config import (
    parse_args,
    build_config,
    get_paths,
    get_device,
    set_seed,
)

from data import (
    load_dataset,
    #load_dataset_with_resampler
)

from training import (
    build_model,
    build_optimizer,
    build_criterion,
    train_loop,
    train_fn,
    model_targ_out,
    model_lhl_out_data,
    bbn_predict,
)

from analysis import (
    nc_analysis,
    nc_analysis_test,
    ohe_classifier,
    ye_classifier_targ,
    evaluate_maj_min_preds,
)

# -----------------------------------------------------------------------------

LOSS_MAP = {
    "ce": "CrossEntropyLoss",
    "mse": "MSELoss",
}

NUM_CLASSES = 10

# ------- frozen weight matrix ------------------------------------------------
def probs(frac):
    """Imbalanced probabilities
    """
    probs = np.array(5 * [1.] + 5 * [frac])
    probs = probs / probs.sum()
    
    return probs 



def householder_V(probs):
    """Householder V matrix for the SVD of h_pi
    """
    v = (np.sqrt(probs) + np.array(9 * [0.] + [1.])).reshape(-1, 1)
    V = (2. / (v.T @ v)) * (v @ v.T) - np.eye(len(probs))
    
    return V



def h_pi(probs):
    """H_pi matrix from probs.
    
    Good for checking that froW bwloNot used.
    """
    H_pi = np.eye(10) - np.sqrt(probs).reshape(-1, 1) @ np.sqrt(probs).reshape(-1, 1).T

    return H_pi



def optimal_W(dim, probs):
    """Frozen weight matrix for Y targets
    """
    V = householder_V(probs)
    
    J = np.eye(len(probs))
    J[-1, -1] = 0.
    
    U = np.eye(dim)[ : , : len(probs)]
    
    print('\n' + 10 * '.' + ' checking optimal weight matrix')
    print('U.T @ U:', np.allclose(U.T @ U, np.eye(len(probs))))
    print('V @ V.T:', np.allclose(V @ V.T, np.eye(len(probs))))
    print('diag J', np.diagonal(J))
    print('h_pi svd', np.allclose(V @ J @ V.T, h_pi(probs)), '\n')
    #print(np.linalg.norm(U @ J @ V.T))
    
    return U @ J @ V.T


# -----------------------------------------------------------------------------

def run_experiment(
    cfg,
    train_loader,
    #train_loader_resampled,
    test_loader,
    device,
):
    #print(cfg)
    
    return main(
        epochs=cfg.epochs,
        train_loader=train_loader,
        #train_loader_resampled=train_loader_resampled,
        test_loader=test_loader,
        loss_name=LOSS_MAP[cfg.loss],
        encoding=cfg.encoding,
        batch_size=cfg.batch_size,
        optimizer_str=cfg.optimizer,
        lrate_factor=cfg.lrate_factor,
        weight_decay=cfg.weight_decay,
        #frozen_weights=cfg.frozen_weights,
        #resampling_factor=cfg.resampling_factor,
        init_noise=cfg.init_noise,
        device=device,
    )


def main(epochs, train_loader, #train_loader_resampled, 
        test_loader, loss_name, encoding, 
        batch_size, optimizer_str, lrate_factor, 
        weight_decay,
        #resampling_factor, 
        #frozen_weights,
        init_noise,
        device):
    """Builds model and optimizer and trains it.
    Model weights are those after init resnet if init_noise < 0.
    Else random normal noise is added to a theoretical optimal fc.weight with 
    std init_noise that of the optimal fc.weight and to fc.bias solution 
    with init_noise std.
    """
    if cfg.init_noise >= 0.:
        pr = probs(cfg.frac)
        
        lhl_weights = optimal_W(512, pr).T
        lhl_bias = np.zeros(len(pr)) #.reshape(-1, 1)
        
        #add random weight noise with std = init_noise * fc.weight.std()
        rr = np.random.rand(*lhl_weights.shape) - 0.5
        rr = rr / rr.std()
        lhl_weights += init_noise * lhl_weights.std() * rr
        
        print('initial weight norm', np.linalg.norm(lhl_weights))
        
        #add random bias noise with std = init_noise 
        bb = np.random.rand(*lhl_bias.shape) - 0.5
        bb = bb / bb.std()
        lhl_bias += init_noise * bb 
        
        #print('w. b shapes', lhl_weights.shape, lhl_bias.shape)
    
    else:
        lhl_bias = None
        lhl_weights = None
        
    model = build_model(num_classes=NUM_CLASSES,
                input_channels=1,
                #frozen_weights=frozen_weights,
                lhl_weights=lhl_weights,
                lhl_bias=lhl_bias,
                device=device)

    optimizer, scheduler = build_optimizer(model,
                                        loss_name=loss_name,
                                        optimizer_name=optimizer_str,
                                        lr_factor=lrate_factor,
                                        weight_decay=weight_decay,
                                        epochs=epochs,
                                        #frozen_weights=frozen_weights
                                        )

    criterion = build_criterion(loss_name)
    mse_history = train_loop(model,
               train_loader,
               #train_loader_resampled,
               test_loader,
               criterion,
               optimizer,
               scheduler,
               epochs,
               encoding,
               device,
               train_fn,
               model_targ_out,
               bbn_predict,
               ye_classifier_targ)

    train_results = model_lhl_out_data(model, train_loader, device, layer='avgpool')
    test_results = model_lhl_out_data(model, test_loader, device, layer='avgpool')
    
    return train_results, test_results, mse_history


# ==========================================================
# Saving
# ==========================================================

def save_experiment_results(
    train_results,
    test_results,
    mse_history,
    cfg,
    results_dir,
    rep_number,
):

    bf_name = Path(
        cfg.bunch_file
    ).stem

    prefix = (
        f"{bf_name}_"
        f"{cfg.encoding}_"
        f"{cfg.frac}_"
        f"{cfg.optimizer}_"
        f"{cfg.batch_size}_"
        f"{cfg.lrate_factor}_"
        f"{cfg.weight_decay}_"
        f"{cfg.epochs}_"
        f"{rep_number}_"
    )

    print("\nsaving results at ", results_dir)

    joblib.dump(
        train_results,
        results_dir + prefix +
        "train_results.joblib",
    )

    joblib.dump(
        test_results,
        results_dir + prefix +
        "test_results.joblib",
    )

    joblib.dump(
        mse_history,
        results_dir + prefix +
        "l_mse.joblib",
    )

    # Save experiment configuration

    joblib.dump(
        asdict(cfg),
        results_dir + prefix +
        "config.joblib",
    )


def dump_mse(
    mse_history,
    cfg,
    results_dir='./',
    rep_number=0,
):

    prefix = (
        #f"{bf_name}_"
        f"{cfg.encoding}_"
        f"{cfg.frac}_"
        f"{cfg.optimizer}_"
        f"{cfg.batch_size}_"
        f"{cfg.lrate_factor}_"
        f"{cfg.weight_decay}_"
        f"{cfg.epochs}_"
        f"{rep_number}_"
    )

    print("\nsaving mse_history as ", results_dir + prefix + "l_mse.joblib")

    joblib.dump(
        mse_history,
        results_dir + prefix + "l_mse.joblib",
    )

    # Save experiment configuration

    
    
# ==========================================================
# Main script
# ==========================================================

if __name__ == "__main__":

    args = parse_args()
    
    cfg = build_config(args)

    set_seed(cfg.seed)
    
    print("\nExperiment configuration:")
    print(cfg, flush=True)

    data_dir, results_dir = get_paths(cfg)

    train_loader, test_loader = load_dataset(
    #train_loader, _ = load_dataset(
        cfg,
        data_dir,
    )
    
    #train_loader_resampled, test_loader = load_dataset_with_resampler(
    #    cfg,
    #    data_dir,
    #)

    device = get_device(cfg)

    t1 = dt.datetime.now(dt.UTC)

    print(
        "\n"
        + "_" * 20
        + " starting at "
        + str(t1)
    )

    for rep in range(cfg.reps):
        
        set_seed(cfg.seed + rep)

        print(
            "\n"
            + "." * 20
            + f" running repetition {rep}"
        )

        (
            train_results,
            test_results,
            mse_history,
        ) = run_experiment(
            cfg,
            train_loader,
            #train_loader_resampled,
            test_loader,
            device,
        )

        # -------------------------
        # Train analysis
        # -------------------------

        (
            targs_out,
            model_out,
            lhl_out,
            w_b_lhl,
        ) = train_results

        print(
            "\ntrain nc_analysis "
            + "." * 20
        )

        nc_analysis(
            lhl_out,
            w_b_lhl,
            targs_out,
            model_out,
            #verbose=True
        )

        # -------------------------
        # Test analysis
        # -------------------------

        (
            targs_out_ts,
            model_out_ts,
            lhl_out_ts,
            _,
        ) = test_results

        print(
            "\ntest nc_analysis "
            + "." * 20
        )

        nc_analysis_test(
            lhl_out,
            targs_out,
            lhl_out_ts,
            targs_out_ts,
            model_out_ts,
            encoding=cfg.encoding,
            frac=cfg.frac
        )
        
        maj_acc_test, min_acc_test = evaluate_maj_min_preds(
            targs_out_ts,
            model_out_ts,
            cfg.frac,
            cfg.encoding,
            #ye_classifier_targ
        )
        
        print(f'..... final test majority acc. {maj_acc_test:.4f}')
        print(f'..... final test minority acc. {min_acc_test:.4f}',
              flush=True)

        pr = probs(cfg.frac)
        H_pi = h_pi(pr)
        w_final = w_b_lhl[0]
        #print('weight_diff_norm', np.linalg.norm(w_final @ w_final.T - H_pi)**2.) 
        wdn = np.linalg.norm(w_final @ w_final.T - H_pi)
        print(f'..... final weight_diff_norm. {wdn:.4f}')
        
        # -------------------------
        # Save results
        # -------------------------

        if cfg.save_results:

            save_experiment_results(
                train_results,
                test_results,
                mse_history,
                cfg,
                results_dir,
                rep,
            )
            
        elif cfg.epochs >= 350:
            dump_mse(
                mse_history,
                cfg,
                results_dir='./exps/',
                rep_number=0,
            )

    t2 = dt.datetime.now(dt.UTC)

    print(
        "\n"
        + "_" * 20
        + " ending at "
        + str(t2)
    )

    print(
        "\ntotal_time:",
        t2 - t1,
    )