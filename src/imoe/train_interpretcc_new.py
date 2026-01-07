"""
CORRECT Enhanced main script with Comet ML support - Does NOT modify InterpretCC or any backbone
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

# Try to import Comet ML
try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("⚠ Comet ML not installed. Install with: pip install comet_ml")

# Import UNMODIFIED InterpretCC
from src.common.fusion_models.interpretcc import InterpretCC

# Import enhanced training (handles decomposition externally)
try:
    from src.imoe.imoe_train import train_and_evaluate_imoe
    ENHANCED_AVAILABLE = True
    print("✓ Able to use ENHANCED training with information decomposition support")
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
    
    # ===== COMET ML ARGUMENTS =====
    parser.add_argument(
        "--use_comet", type=str2bool, default=False,
        help="Enable Comet ML logging"
    )
    parser.add_argument(
        "--comet_api_key", type=str, default="Fd1aGmcly8SdDO5Ez4DMyCIt5",
        help="Comet ML API key (or set COMET_API_KEY env var)"
    )
    parser.add_argument(
        "--comet_project", type=str, default="imoe-interpretcc",
        help="Comet ML project name"
    )
    parser.add_argument(
        "--comet_workspace", type=str, default="mattam301",
        help="Comet ML workspace (optional)"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Custom experiment name for Comet ML"
    )
    parser.add_argument(
        "--comet_tags", type=str, nargs='+', default=None,
        help="Tags for Comet ML experiment"
    )
    
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


def init_comet_experiment(args):
    """Initialize Comet ML experiment if enabled"""
    if not args.use_comet:
        return None
    
    if not COMET_AVAILABLE:
        print("⚠ Comet ML requested but not installed. Continuing without logging.")
        return None
    
    # Get API key from args or environment
    api_key = args.comet_api_key or os.environ.get('COMET_API_KEY')
    
    if not api_key:
        print("⚠ Comet ML API key not provided. Set COMET_API_KEY env var or use --comet_api_key")
        return None
    
    # Create experiment
    experiment = Experiment(
        api_key=api_key,
        project_name=args.comet_project,
        workspace=args.comet_workspace,
        auto_metric_logging=True,
        auto_param_logging=True,
        auto_output_logging="simple",
    )
    
    # Set experiment name
    if args.experiment_name:
        experiment.set_name(args.experiment_name)
    else:
        # Auto-generate name based on config
        name = f"{args.data}_{args.modality}_lr{args.lr}_bs{args.batch_size}"
        if ENHANCED_AVAILABLE and getattr(args, 'use_info_decomposition', False):
            name += "_decomposed"
        experiment.set_name(name)
    
    # Add tags
    tags = args.comet_tags or []
    tags.extend([args.data, f"modality_{args.modality}", "interpretcc"])
    if ENHANCED_AVAILABLE and getattr(args, 'use_info_decomposition', False):
        tags.append("info_decomposition")
    experiment.add_tags(tags)
    
    # Log all hyperparameters
    experiment.log_parameters(vars(args))
    
    print(f"✓ Comet ML experiment initialized: {experiment.url}")
    
    return experiment


def log_run_metrics(experiment, seed, run_idx, metrics, stage="test"):
    """Log metrics for a single run to Comet ML"""
    if experiment is None:
        return
    
    with experiment.context_manager(f"run_{run_idx}_seed_{seed}"):
        for metric_name, metric_value in metrics.items():
            experiment.log_metric(f"{stage}_{metric_name}", metric_value)


def log_summary_metrics(experiment, summary_metrics):
    """Log summary metrics across all runs to Comet ML"""
    if experiment is None:
        return
    
    with experiment.context_manager("summary"):
        for metric_name, metric_value in summary_metrics.items():
            experiment.log_metric(metric_name, metric_value)


def main():
    args, _ = parse_args()
    
    # Initialize Comet ML experiment
    experiment = init_comet_experiment(args)
    
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
    
    # Log model configuration to Comet
    if experiment:
        experiment.log_parameters(model_kwargs)
    
    # Add decomposition config if available and enabled
    use_decomposition = ENHANCED_AVAILABLE and getattr(args, 'use_info_decomposition', False)
    
    if use_decomposition:
        model_kwargs["use_info_decomposition"] = True
        model_kwargs["decomposition_alpha_u"] = args.decomposition_alpha_u
        model_kwargs["decomposition_alpha_r"] = args.decomposition_alpha_r
        model_kwargs["decomposition_alpha_s"] = args.decomposition_alpha_s
        model_kwargs["decomposition_loss_weight"] = args.decomposition_loss_weight
        
        # Log decomposition params to Comet
        if experiment:
            decomp_params = {
                "decomposition_enabled": True,
                "decomposition_alpha_u": args.decomposition_alpha_u,
                "decomposition_alpha_r": args.decomposition_alpha_r,
                "decomposition_alpha_s": args.decomposition_alpha_s,
                "decomposition_loss_weight": args.decomposition_loss_weight,
            }
            experiment.log_parameters(decomp_params)
        
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
        
        # Log run start to Comet
        if experiment:
            experiment.log_text(f"Starting run {i+1} with seed {seed}")
        
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
            
            # Log run metrics to Comet
            log_run_metrics(experiment, seed, i, {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "test_mae": test_mae,
                "train_time": train_time,
                "infer_time": infer_time,
                "flops": flop,
                "params": param,
            })
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
            
            # Log run metrics to Comet
            log_run_metrics(experiment, seed, i, {
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_f1_micro": test_f1_micro,
                "test_auc": test_auc,
                "train_time": train_time,
                "infer_time": infer_time,
                "flops": flop,
                "params": param,
            })
        
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

    # Log computational metrics to Comet
    if experiment:
        computational_metrics = {
            "mean_train_time": mean_train_time,
            "std_train_time": np.sqrt(variance_train_time),
            "mean_infer_time": mean_infer_time,
            "std_infer_time": np.sqrt(variance_infer_time),
            "mean_flops": mean_flop,
            "std_flops": np.sqrt(variance_flop),
            "mean_gflops": mean_gflop,
            "std_gflops": np.sqrt(variance_gflop),
            "mean_params": mean_param,
            "std_params": np.sqrt(variance_param),
        }
        log_summary_metrics(experiment, computational_metrics)

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
        
        # Log final metrics to Comet
        if experiment:
            final_metrics = {
                "val_avg_acc": val_avg_acc,
                "val_std_acc": val_std_acc,
                "val_avg_loss": val_avg_loss,
                "val_std_loss": val_std_loss,
                "test_avg_acc": test_avg_acc,
                "test_std_acc": test_std_acc,
                "test_avg_mae": test_avg_mae,
                "test_std_mae": test_std_mae,
            }
            log_summary_metrics(experiment, final_metrics)

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
        
        # Log final metrics to Comet
        if experiment:
            final_metrics = {
                "val_avg_acc": val_avg_acc,
                "val_std_acc": val_std_acc,
                "val_avg_f1": val_avg_f1,
                "val_std_f1": val_std_f1,
                "val_avg_auc": val_avg_auc,
                "val_std_auc": val_std_auc,
                "test_avg_acc": test_avg_acc,
                "test_std_acc": test_std_acc,
                "test_avg_f1_macro": test_avg_f1,
                "test_std_f1_macro": test_std_f1,
                "test_avg_f1_micro": test_avg_f1_micro,
                "test_std_f1_micro": test_std_f1_micro,
                "test_avg_auc": test_avg_auc,
                "test_std_auc": test_std_auc,
            }
            log_summary_metrics(experiment, final_metrics)

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
    
    # Log summary text to Comet
    if experiment:
        experiment.log_text(log_summary, metadata={"type": "summary"})
    
    print(f"\n{'='*80}")
    print("✓ Training completed successfully!")
    if experiment:
        print(f"✓ Results logged to Comet ML: {experiment.url}")
    print(f"{'='*80}\n")
    
    # End Comet experiment
    if experiment:
        experiment.end()


if __name__ == "__main__":
    main()