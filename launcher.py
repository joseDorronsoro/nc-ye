#!/usr/bin/env python
# coding: utf-8
########################################################################################################
import sys
import os
import joblib
import datetime as dt
from pathlib import Path

from config import (
    parse_args,
    build_config,
    get_paths,
    get_device,
    set_seed,
)

from data import (
    load_dataset,
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


# -----------------------------------------------------------------------------

def run_experiment(
    cfg,
    train_loader,
    test_loader,
    device,
):
    print(cfg)
    
    return main(
        epochs=cfg.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_name=LOSS_MAP[cfg.loss],
        encoding=cfg.encoding,
        batch_size=cfg.batch_size,
        optimizer_str=cfg.optimizer,
        lrate_factor=cfg.lrate_factor,
        device=device,
    )


def main(epochs, train_loader, test_loader, loss_name, encoding, batch_size, optimizer_str, lrate_factor, device):
    """
    """
    model = build_model(num_classes=NUM_CLASSES,
                input_channels=1,
                device=device)

    optimizer, scheduler = build_optimizer(model,
                                        loss_name=loss_name,
                                        optimizer_name=optimizer_str,
                                        lr_factor=lrate_factor)

    criterion = build_criterion(loss_name)
    mse_history = train_loop(model,
               train_loader,
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
        f"{rep_number}_"
    )

    print("\nsaving results ..........")

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


# ==========================================================
# Main script
# ==========================================================

if __name__ == "__main__":

    args = parse_args()

    cfg = build_config(args)

    set_seed(cfg.seed)
    
    print("\nExperiment configuration:")
    print(cfg)

    data_dir, results_dir = get_paths(cfg)

    train_loader, test_loader = load_dataset(
        cfg,
        data_dir,
    )

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
        
        print(f'..... test majority acc. {maj_acc_test:.4f}')
        print(f'..... test minority acc. {min_acc_test:.4f}',
              flush=True)


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