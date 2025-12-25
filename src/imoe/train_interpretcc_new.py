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

from src.common.fusion_models.interpretcc import InterpretCC
from src.imoe.imoe_train import train_and_evaluate_imoe_with_pretrain
from src.common.utils import setup_logger, str2bool


# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="iMoE-interpretcc with Unified Pre-training")
    
    # Data and modality settings
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument("--modality", type=str, default="IGCB")
    parser.add_argument("--patch", type=str2bool, default=False)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--initial_filling", type=str, default="mean")
    
    # Device and reproducibility
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--use_common_ids", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)

    # ============ PHASE 1: PRE-TRAINING DECOMPOSITION ============
    parser.add_argument(
        "--use_pretrain", type=str2bool, default=True,
        help="Whether to pre-train the decomposition module"
    )
    parser.add_argument(
        "--pretrain_epochs", type=int, default=10,
        help="Number of epochs for pre-training decomposition"
    )
    parser.add_argument(
        "--pretrain_lr", type=float, default=1e-3,
        help="Learning rate for pre-training"
    )
    parser.add_argument(
        "--pretrain_method", type=str, default="reconstruction",
        choices=["reconstruction", "contrastive", "mutual_info"],
        help="Pre-training objective for learning U/R/S decomposition"
    )
    parser.add_argument(
        "--pretrain_weight_decay", type=float, default=1e-5,
        help="Weight decay for pre-training"
    )
    
    # ============ PHASE 2: MAIN TRAINING WITH EXPERTS ============
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--freeze_decomposer", type=str2bool, default=False,
        help="Whether to freeze decomposer during main training"
    )
    
    # Reweighting model settings
    parser.add_argument("--temperature_rw", type=float, default=1.0)
    parser.add_argument("--hidden_dim_rw", type=int, default=256)
    parser.add_argument("--num_layer_rw", type=int, default=1)
    
    # Loss weights
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)
    parser.add_argument("--orthogonality_loss_weight", type=float, default=1e-3)
    parser.add_argument("--load_balancing_weight", type=float, default=1e-2)
    
    # Architecture settings
    parser.add_argument("--fusion_sparse", type=str2bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers_enc", type=int, default=1)
    parser.add_argument("--num_layers_fus", type=int, default=1)
    parser.add_argument("--num_layers_pred", type=int, default=1)

    # Gumbel-Softmax settings
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)
    
    # ============ DECOMPOSITION ARCHITECTURE ============
    parser.add_argument(
        "--use_decomposition", type=str2bool, default=True,
        help="Whether to use U/R/S decomposition for expert routing"
    )
    parser.add_argument(
        "--decomposition_method", type=str, default="learned",
        choices=["pidcg", "learned", "attention"],
        help="Method for computing information decomposition"
    )
    parser.add_argument(
        "--num_experts_per_type", type=int, default=2,
        help="Number of experts per interaction type (U/R/S)"
    )
    parser.add_argument(
        "--expert_specialization", type=str2bool, default=True,
        help="Whether to enforce expert specialization"
    )
    
    # Interpretability settings
    parser.add_argument("--log_interactions", type=str2bool, default=True)
    parser.add_argument("--save_interaction_maps", type=str2bool, default=False)

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
    
    # Setup logging
    log_dir = f"./logs/imoe/interpretcc_unified/{args.data}"
    if args.use_pretrain:
        log_dir += f"_pretrain{args.pretrain_epochs}"
    if args.use_decomposition:
        log_dir += f"_{args.decomposition_method}"
    
    logger = setup_logger(log_dir, f"{args.data}", f"{args.modality}.txt")
    
    seeds = np.arange(args.n_runs)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    num_modalities = len(args.modality)

    log_summary = "=" * 90 + "\n"
    log_summary += "UNIFIED iMoE: PRE-TRAIN DECOMPOSITION → TRAIN SPECIALIZED EXPERTS\n"
    log_summary += "=" * 90 + "\n"

    # Model configuration
    model_kwargs = {
        "model": "iMoE-InterpretCC-Unified",
        "phase_1_pretrain": args.use_pretrain,
        "pretrain_epochs": args.pretrain_epochs if args.use_pretrain else 0,
        "pretrain_method": args.pretrain_method if args.use_pretrain else None,
        "pretrain_lr": args.pretrain_lr if args.use_pretrain else None,
        "phase_2_train_epochs": args.train_epochs,
        "freeze_decomposer": args.freeze_decomposer,
        "decomposition_method": args.decomposition_method,
        "num_experts_per_type": args.num_experts_per_type,
        "expert_specialization": args.expert_specialization,
        "modality": args.modality,
        "num_modalities": num_modalities,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
    }

    log_summary += "Configuration:\n"
    for key, value in model_kwargs.items():
        log_summary += f"  {key}: {value}\n"
    log_summary += "\n"

    print(f"\n{'='*70}")
    print("UNIFIED TRAINING PIPELINE")
    print(f"{'='*70}")
    print(f"Dataset: {args.data} | Modality: {args.modality}")
    if args.use_pretrain:
        print(f"Phase 1: Pre-train decomposition ({args.pretrain_method}) for {args.pretrain_epochs} epochs")
        print(f"Phase 2: Train {args.num_experts_per_type} experts/type for {args.train_epochs} epochs")
        print(f"         Freeze decomposer: {args.freeze_decomposer}")
    else:
        print(f"Single-phase training for {args.train_epochs} epochs")
    print(f"{'='*70}\n")

    # Dataset configuration
    data_to_nlabels = {
        "adni": 3, "mimic": 2, "mmimdb": 23, 
        "enrico": 20, "mosi": 2, "mosi_regression": 1,
    }
    n_labels = data_to_nlabels[args.data]

    # Initialize metric containers
    if args.data == "mosi_regression":
        val_losses, val_accs, test_accs, test_maes = [], [], [], []
    else:
        val_accs, val_f1s, val_aucs = [], [], []
        test_accs, test_f1s, test_f1_micros, test_aucs = [], [], [], []

    # Efficiency and interpretability metrics
    pretrain_losses, pretrain_times = [], []
    train_times, infer_times, flops, params = [], [], [], []
    
    if args.log_interactions:
        unique_ratios, redundant_ratios, synergistic_ratios = [], [], []
        expert_utilizations = []

    # ============ MAIN TRAINING LOOP ============
    for run_idx, seed in enumerate(seeds):
        print(f"\n{'='*70}")
        print(f"Run {run_idx + 1}/{len(seeds)} | Seed {seed}")
        print(f"{'='*70}")
        
        # Initialize fusion model
        fusion_model = InterpretCC(
            num_classes=n_labels,
            num_modality=num_modalities,
            input_dim=args.hidden_dim,
            dropout=args.dropout,
            tau=args.tau,
            hard=args.hard,
            threshold=args.threshold,
            use_decomposition=args.use_decomposition,
            decomposition_method=args.decomposition_method if args.use_decomposition else None,
            num_experts_per_type=args.num_experts_per_type if args.use_decomposition else 1,
            expert_specialization=args.expert_specialization if args.use_decomposition else False,
        ).to(device)

        # ============ UNIFIED TRAINING: BOTH PHASES IN ONE CALL ============
        results = train_and_evaluate_imoe_with_pretrain(
            args=args,
            seed=seed,
            fusion_model=fusion_model,
            model_name="interpretcc_unified",
            device=device,
        )
        
        # Unpack results
        if args.data == "mosi_regression":
            (
                val_loss, val_acc, test_acc, test_mae,
                train_time, infer_time, flop, param,
                pretrain_loss, pretrain_time,
                *extra_metrics
            ) = results
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_maes.append(test_mae)
        else:
            (
                val_acc, val_f1, val_auc,
                test_acc, test_f1, test_f1_micro, test_auc,
                train_time, infer_time, flop, param,
                pretrain_loss, pretrain_time,
                *extra_metrics
            ) = results
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            val_aucs.append(val_auc)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)
            test_f1_micros.append(test_f1_micro)
            test_aucs.append(test_auc)
        
        # Store metrics
        train_times.append(train_time)
        infer_times.append(infer_time)
        flops.append(flop)
        params.append(param)
        
        if args.use_pretrain:
            pretrain_losses.append(pretrain_loss)
            pretrain_times.append(pretrain_time)
        
        # Interpretability metrics
        if args.log_interactions and len(extra_metrics) >= 4:
            unique_ratios.append(extra_metrics[0])
            redundant_ratios.append(extra_metrics[1])
            synergistic_ratios.append(extra_metrics[2])
            expert_utilizations.append(extra_metrics[3])

    # ============ AGGREGATE RESULTS ============
    log_summary += "\n" + "=" * 90 + "\n"
    log_summary += "RESULTS\n"
    log_summary += "=" * 90 + "\n\n"
    
    # Pre-training metrics
    if args.use_pretrain and pretrain_losses:
        log_summary += "PHASE 1: Pre-training Decomposition\n"
        log_summary += "-" * 90 + "\n"
        mean_pt_loss = np.mean(pretrain_losses)
        std_pt_loss = np.std(pretrain_losses)
        mean_pt_time = np.mean(pretrain_times)
        std_pt_time = np.std(pretrain_times)
        
        log_summary += f"Pre-train Loss:  {mean_pt_loss:.4f} ± {std_pt_loss:.4f}\n"
        log_summary += f"Pre-train Time:  {mean_pt_time:.2f} ± {std_pt_time:.2f} seconds\n\n"
    
    # Main training efficiency
    log_summary += "PHASE 2: Training Specialized Experts\n"
    log_summary += "-" * 90 + "\n"
    
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times)
    mean_infer_time = np.mean(infer_times)
    std_infer_time = np.std(infer_times)
    mean_param = np.mean(params)
    std_param = np.std(params)
    mean_gflop = np.mean(np.array(flops) / 1e9)
    std_gflop = np.std(np.array(flops) / 1e9)

    log_summary += f"Train time/epoch:     {mean_train_time:.2f} ± {std_train_time:.2f} s\n"
    log_summary += f"Inference time/epoch: {mean_infer_time:.2f} ± {std_infer_time:.2f} s\n"
    log_summary += f"Parameters:           {mean_param:,.0f} ± {std_param:,.0f}\n"
    log_summary += f"GFLOPs:               {mean_gflop:.2f} ± {std_gflop:.2f}\n\n"

    # Performance metrics
    log_summary += "Performance Metrics\n"
    log_summary += "-" * 90 + "\n"
    
    if args.data == "mosi_regression":
        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_loss = np.mean(val_losses)
        val_std_loss = np.std(val_losses)
        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_mae = np.mean(test_maes)
        test_std_mae = np.std(test_maes)

        log_summary += f"[Val]  Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f}%\n"
        log_summary += f"[Val]  Loss:     {val_avg_loss:.4f} ± {val_std_loss:.4f}\n"
        log_summary += f"[Test] Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}%\n"
        log_summary += f"[Test] MAE:      {test_avg_mae:.4f} ± {test_std_mae:.4f}\n"

        print(f"\n{'='*70}")
        print("FINAL RESULTS (Regression)")
        print(f"{'='*70}")
        print(f"Val:  Acc={val_avg_acc:.2f}±{val_std_acc:.2f}% | Loss={val_avg_loss:.4f}±{val_std_loss:.4f}")
        print(f"Test: Acc={test_avg_acc:.2f}±{test_std_acc:.2f}% | MAE={test_avg_mae:.4f}±{test_std_mae:.4f}")
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

        log_summary += f"[Val]  Accuracy:       {val_avg_acc:.2f} ± {val_std_acc:.2f}%\n"
        log_summary += f"[Val]  F1 (Macro):     {val_avg_f1:.2f} ± {val_std_f1:.2f}%\n"
        log_summary += f"[Val]  AUC:            {val_avg_auc:.2f} ± {val_std_auc:.2f}%\n"
        log_summary += f"[Test] Accuracy:       {test_avg_acc:.2f} ± {test_std_acc:.2f}%\n"
        log_summary += f"[Test] F1 (Macro):     {test_avg_f1:.2f} ± {test_std_f1:.2f}%\n"
        log_summary += f"[Test] F1 (Micro):     {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f}%\n"
        log_summary += f"[Test] AUC:            {test_avg_auc:.2f} ± {test_std_auc:.2f}%\n"

        print(f"\n{'='*70}")
        print("FINAL RESULTS (Classification)")
        print(f"{'='*70}")
        print(f"Val:  Acc={val_avg_acc:.2f}±{val_std_acc:.2f}% | F1={val_avg_f1:.2f}±{val_std_f1:.2f}% | AUC={val_avg_auc:.2f}±{val_std_auc:.2f}%")
        print(f"Test: Acc={test_avg_acc:.2f}±{test_std_acc:.2f}% | F1={test_avg_f1:.2f}±{test_std_f1:.2f}% | AUC={test_avg_auc:.2f}±{test_std_auc:.2f}%")

    # Interpretability metrics
    if args.log_interactions and unique_ratios:
        log_summary += "\n"
        log_summary += "Interpretability: Information Decomposition\n"
        log_summary += "-" * 90 + "\n"
        
        mean_unique = np.mean(unique_ratios) * 100
        std_unique = np.std(unique_ratios) * 100
        mean_redundant = np.mean(redundant_ratios) * 100
        std_redundant = np.std(redundant_ratios) * 100
        mean_synergistic = np.mean(synergistic_ratios) * 100
        std_synergistic = np.std(synergistic_ratios) * 100
        
        log_summary += f"Unique Information:       {mean_unique:.2f} ± {std_unique:.2f}%\n"
        log_summary += f"Redundant Information:    {mean_redundant:.2f} ± {std_redundant:.2f}%\n"
        log_summary += f"Synergistic Information:  {mean_synergistic:.2f} ± {std_synergistic:.2f}%\n"
        
        if expert_utilizations:
            mean_util = np.mean(expert_utilizations, axis=0)
            log_summary += f"\nExpert Utilization:\n"
            for i, util in enumerate(mean_util):
                expert_type = ["Unique", "Redundant", "Synergistic"][i // args.num_experts_per_type]
                expert_num = i % args.num_experts_per_type + 1
                log_summary += f"  {expert_type} Expert {expert_num}: {util:.2%}\n"
        
        print(f"\nInformation Decomposition:")
        print(f"  Unique:      {mean_unique:.2f}±{std_unique:.2f}%")
        print(f"  Redundant:   {mean_redundant:.2f}±{std_redundant:.2f}%")
        print(f"  Synergistic: {mean_synergistic:.2f}±{std_synergistic:.2f}%")

    log_summary += "\n" + "=" * 90 + "\n"
    logger.info(log_summary)
    
    print(f"\n{'='*70}")
    print("✓ Training completed successfully!")
    print(f"✓ Logs saved to: {log_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()