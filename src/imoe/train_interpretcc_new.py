"""
CORRECT Enhanced main script - Does NOT modify InterpretCC or any backbone
Path: src/imoe/train_interpretcc_new.py

Key principle: All decomposition happens EXTERNALLY in the training wrapper.
InterpretCC and all existing models remain 100% unchanged.
"""

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import numpy as np
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

# Import UNMODIFIED InterpretCC
from src.common.fusion_models.interpretcc import InterpretCC

# Import enhanced training (handles decomposition externally)
try:
    from src.imoe.imoe_train import train_and_evaluate_imoe
    ENHANCED_AVAILABLE = True
    print("✓ Using ENHANCED training with information decomposition support")
except ImportError:
    from src.imoe.imoe_train import train_and_evaluate_imoe
    ENHANCED_AVAILABLE = False
    print("⚠ Enhanced module not found, using original training")

from src.common.utils import setup_logger, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="iMoE-interpretcc")
    
    # ===== ORIGINAL ARGUMENTS (All work exactly as before) =====
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument("--modality", type=str, default="IGCB")
    parser.add_argument("--patch", type=str2bool, default=False)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--initial_filling", type=str, default="mean")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--use_common_ids", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature_rw", type=float, default=1)
    parser.add_argument("--hidden_dim_rw", type=int, default=256)
    parser.add_argument("--num_layer_rw", type=int, default=1)
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)
    parser.add_argument("--fusion_sparse", type=str2bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers_enc", type=int, default=1)
    parser.add_argument("--num_layers_fus", type=int, default=1)
    parser.add_argument("--num_layers_pred", type=int, default=1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gate_loss_weight", type=float, default=1e-2)
    
    # ===== NEW ARGUMENTS (Only if enhanced module available) =====
    if ENHANCED_AVAILABLE:
        parser.add_argument(
            "--use_info_decomposition", type=str2bool, default=False,
            help="Enable information-theoretic decomposition (external to InterpretCC)"
        )
        parser.add_argument(
            "--decomposition_alpha_u", type=float, default=1.0,
            help="Weight for uniqueness loss"
        )
        parser.add_argument(
            "--decomposition_alpha_r", type=float, default=1.0,
            help="Weight for redundancy loss"
        )
        parser.add_argument(
            "--decomposition_alpha_s", type=float, default=0.5,
            help="Weight for synergy loss"
        )
        parser.add_argument(
            "--decomposition_loss_weight", type=float, default=0.01,
            help="Overall weight for decomposition loss"
        )

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    
    logger = setup_logger(
        f"./logs/imoe/interpretcc/{args.data}",
        f"{args.data}",
        f"{args.modality}.txt",
    )
    
    seeds = np.arange(args.n_runs)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_modalities = len(args.modality)

    log_summary = "======================================================================================\n"

    # Model configuration
    model_kwargs = {
        "model": "Interaction-MoE-interpretcc",
        "temperature_rw": args.temperature_rw,
        "hidden_dim_rw": args.hidden_dim_rw,
        "num_layer_rw": args.num_layer_rw,
        "interaction_loss_weight": args.interaction_loss_weight,
        "modality": args.modality,
        "tau": args.tau,
        "hard": args.hard,
        "threshold": args.threshold,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
    }
    
    # Add decomposition config if available and enabled
    use_decomposition = ENHANCED_AVAILABLE and getattr(args, 'use_info_decomposition', False)
    
    if use_decomposition:
        model_kwargs["use_info_decomposition"] = True
        model_kwargs["decomposition_alpha_u"] = args.decomposition_alpha_u
        model_kwargs["decomposition_alpha_r"] = args.decomposition_alpha_r
        model_kwargs["decomposition_alpha_s"] = args.decomposition_alpha_s
        model_kwargs["decomposition_loss_weight"] = args.decomposition_loss_weight
        
        log_summary += "\n" + "="*80 + "\n"
        log_summary += "INFORMATION DECOMPOSITION ENABLED (External to InterpretCC)\n"
        log_summary += f"  Alpha U (Uniqueness): {args.decomposition_alpha_u}\n"
        log_summary += f"  Alpha R (Redundancy): {args.decomposition_alpha_r}\n"
        log_summary += f"  Alpha S (Synergy): {args.decomposition_alpha_s}\n"
        log_summary += f"  Decomposition Loss Weight: {args.decomposition_loss_weight}\n"
        log_summary += "  NOTE: Decomposition happens in preprocessing, InterpretCC unchanged\n"
        log_summary += "="*80 + "\n"

    log_summary += f"\nModel configuration: {model_kwargs}\n"
    print("Modality:", args.modality)

    data_to_nlabels = {
        "adni": 3,
        "mimic": 2,
        "mmimdb": 23,
        "enrico": 20,
        "mosi": 2,
        "mosi_regression": 1,
    }
    n_labels = data_to_nlabels[args.data]

    # Initialize metric lists
    if args.data == "mosi_regression":
        val_losses = []
        val_accs = []
        test_accs = []
        test_maes = []
    else:
        val_accs = []
        val_f1s = []
        val_aucs = []
        test_accs = []
        test_f1s = []
        test_f1_micros = []
        test_aucs = []

    train_times = []
    infer_times = []
    flops = []
    params = []

    # Training loop
    for i, seed in enumerate(seeds if len(seeds) > 1 else [args.seed]):
        print(f"\n{'='*80}")
        print(f"Run {i+1}/{len(seeds) if len(seeds) > 1 else 1}, Seed: {seed}")
        print(f"{'='*80}\n")
        
        # ===== CRITICAL: InterpretCC is initialized with ORIGINAL signature only =====
        # NO new arguments added to InterpretCC - it remains completely unchanged
        fusion_model = InterpretCC(
            num_classes=n_labels,
            num_modality=len(args.modality),
            input_dim=args.hidden_dim,
            dropout=args.dropout,
            tau=args.tau,
            hard=args.hard,
            threshold=args.threshold,
        ).to(device)
        
        if use_decomposition:
            print("Note: Decomposition preprocessing will be applied externally")
            print("      InterpretCC remains unmodified\n")

        # Train and evaluate (decomposition handled inside train_and_evaluate_imoe)
        if args.data == "mosi_regression":
            (
                val_loss, val_acc, test_acc, test_mae,
                train_time, infer_time, flop, param,
            ) = train_and_evaluate_imoe(args, seed, fusion_model, "interpretcc")
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_maes.append(test_mae)
        else:
            (
                val_acc, val_f1, val_auc,
                test_acc, test_f1, test_f1_micro, test_auc,
                train_time, infer_time, flop, param,
            ) = train_and_evaluate_imoe(args, seed, fusion_model, "interpretcc")
            
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_f1_micros.append(test_f1_micro)
            test_aucs.append(test_auc)
        
        train_times.append(train_time)
        infer_times.append(infer_time)
        flops.append(flop)
        params.append(param)

    # Compute statistics
    mean_train_time = np.mean(train_times)
    variance_train_time = np.var(train_times)
    mean_infer_time = np.mean(infer_times)
    variance_infer_time = np.var(infer_times)
    mean_flop = np.mean(flops)
    variance_flop = np.var(flops)
    mean_gflop = np.mean(np.array(flops) / 1e9)
    variance_gflop = np.var(np.array(flops) / 1e9)
    mean_param = np.mean(params)
    variance_param = np.var(params)

    log_summary += "\n"
    log_summary += f"Train one epoch time: {mean_train_time:.2f} ± {variance_train_time:.2f}\n"
    log_summary += f"Inference one epoch time: {mean_infer_time:.2f} ± {variance_infer_time:.2f}\n"
    log_summary += f"flops: {mean_flop:,.0f} ± {variance_flop:,.0f}\n"
    log_summary += f"gflops: {mean_gflop:.2f} ± {variance_gflop:.2f}\n"
    log_summary += f"param: {mean_param:,.0f} ± {variance_param:,.0f}\n"

    # Results
    if args.data == "mosi_regression":
        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_loss = np.mean(val_losses)
        val_std_loss = np.std(val_losses)
        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_mae = np.mean(test_maes)
        test_std_mae = np.std(test_maes)

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f}\n"
        log_summary += f"[Val] Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f}\n"
        log_summary += f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}\n"
        log_summary += f"[Test] Mean Absolute Error: {test_avg_mae:.2f} ± {test_std_mae:.2f}\n"

        print(model_kwargs)
        print(f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f}")
        print(f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}")
    else:
        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_f1 = np.mean(val_f1s) * 100
        val_std_f1 = np.std(val_f1s) * 100
        val_avg_auc = np.mean(val_aucs) * 100
        val_std_auc = np.std(val_aucs) * 100

        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_f1 = np.mean(test_f1s) * 100
        test_std_f1 = np.std(test_f1s) * 100
        test_avg_f1_micro = np.mean(test_f1_micros) * 100
        test_std_f1_micro = np.std(test_f1_micros) * 100
        test_avg_auc = np.mean(test_aucs) * 100
        test_std_auc = np.std(test_aucs) * 100

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f}\n"
        log_summary += f"[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f}\n"
        log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}\n"
        log_summary += f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}\n"
        log_summary += f"[Test] Average F1 (Macro) Score: {test_avg_f1:.2f} ± {test_std_f1:.2f}\n"
        log_summary += f"[Test] Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f}\n"
        log_summary += f"[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}\n"

        print(model_kwargs)
        print(f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}")
        print(f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}")

    logger.info(log_summary)
    
    print(f"\n{'='*80}")
    print("✓ Training completed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()