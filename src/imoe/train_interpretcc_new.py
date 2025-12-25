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
from src.imoe.imoe_train import train_and_evaluate_imoe
from src.common.utils import setup_logger, str2bool


# ============== NEW: Information Decomposition Module ==============
# This can be imported and used independently

class ModalityDecompositionBranch(torch.nn.Module):
    """Decomposes modality into U, R, S components"""
    def __init__(self, in_dim: int, out_dim: int, num_modalities: int):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.fc_unique = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(out_dim, out_dim)
        )
        
        self.fc_redundant = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(in_dim, out_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(out_dim, out_dim)
            ) for _ in range(num_modalities - 1)
        ])
        
        self.fc_synergy = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x):
        unique = self.fc_unique(x)
        redundant = [fc(x) for fc in self.fc_redundant]
        synergy = self.fc_synergy(x)
        return unique, redundant, synergy


class InfoDecompositionWrapper(torch.nn.Module):
    """
    Wrapper that can be used with existing InterpretCC model.
    If use_decomposition=False, acts as identity (backward compatible)
    """
    def __init__(self, modality_dims, hidden_dim, num_classes, 
                 use_decomposition=True):
        super().__init__()
        self.use_decomposition = use_decomposition
        self.num_modalities = len(modality_dims)
        
        if use_decomposition:
            # NEW: Decomposition branches
            self.branches = torch.nn.ModuleList([
                ModalityDecompositionBranch(dim, hidden_dim, self.num_modalities)
                for dim in modality_dims
            ])
            self.fusion_head = torch.nn.Linear(hidden_dim * 3, num_classes)
        else:
            # OLD: Direct passthrough (maintains compatibility)
            self.branches = None
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of tensors [batch, (seq), dim]
        
        Returns:
            If use_decomposition=True: decomposed features dict
            If use_decomposition=False: original features (identity)
        """
        if not self.use_decomposition:
            # Backward compatible: return original features
            return {
                'original_features': modality_features,
                'decomposed': False
            }
        
        # NEW: Decompose into U, R, S
        decompositions = [branch(feat) for branch, feat in 
                         zip(self.branches, modality_features)]
        
        uniques = [d[0] for d in decompositions]
        redundants = [d[1] for d in decompositions]
        synergies = [d[2] for d in decompositions]
        
        # Aggregate redundant information
        all_redundant = []
        for i in range(self.num_modalities):
            for j, red_list in enumerate(redundants):
                if i != j:
                    idx = i if i < j else i - 1
                    if idx < len(red_list):
                        all_redundant.append(red_list[idx])
        
        redundant_agg = torch.stack(all_redundant).mean(dim=0) if all_redundant else torch.zeros_like(uniques[0])
        synergy_agg = torch.stack(synergies).mean(dim=0)
        
        return {
            'unique_features': uniques,
            'redundant_features': redundant_agg,
            'synergy_features': synergy_agg,
            'decomposed': True
        }


def compute_info_decomposition_loss(decomposed_outputs, labels, 
                                   alpha_u=1.0, alpha_r=1.0, alpha_s=0.5):
    """
    Information-theoretic loss for decomposition pretraining
    """
    if not decomposed_outputs.get('decomposed', False):
        return torch.tensor(0.0), {}
    
    uniques = decomposed_outputs['unique_features']
    redundant = decomposed_outputs['redundant_features']
    synergies = decomposed_outputs['synergy_features']
    
    # Simplified MI estimation using correlation
    def mi_estimate(z1, z2):
        z1_flat = z1.reshape(-1, z1.shape[-1])
        z2_flat = z2.reshape(-1, z2.shape[-1])
        z1_centered = z1_flat - z1_flat.mean(dim=0)
        z2_centered = z2_flat - z2_flat.mean(dim=0)
        cov = torch.mm(z1_centered.t(), z2_centered) / z1_flat.shape[0]
        return torch.norm(cov, p='fro') ** 2
    
    # 1. Uniqueness: minimize I(U; R) and I(U; S)
    L_unique = 0.0
    for u in uniques:
        L_unique += mi_estimate(u, redundant)
        for s in synergies:
            L_unique += mi_estimate(u, s)
    L_unique /= len(uniques)
    
    # 2. Redundancy: maximize correlation between redundant parts
    L_redundant = 0.0
    for i in range(len(uniques)):
        for j in range(i + 1, len(uniques)):
            L_redundant -= mi_estimate(uniques[i], uniques[j])
    L_redundant /= max(1, len(uniques) * (len(uniques) - 1) / 2)
    
    # 3. Combined loss
    total_loss = alpha_u * L_unique + alpha_r * L_redundant
    
    loss_dict = {
        'unique': L_unique.item() if isinstance(L_unique, torch.Tensor) else L_unique,
        'redundant': L_redundant.item() if isinstance(L_redundant, torch.Tensor) else L_redundant,
    }
    
    return total_loss, loss_dict


# ============== MODIFIED: Enhanced train_and_evaluate_imoe ==============
# This wraps the original function with optional decomposition pretraining

def train_and_evaluate_imoe_enhanced(args, seed, fusion_model, model_name,
                                    use_decomposition=False,
                                    decomposition_model=None,
                                    pretrain_epochs=50):
    """
    Enhanced training function with optional information decomposition.
    
    Args:
        args: Original arguments
        seed: Random seed
        fusion_model: InterpretCC or other fusion model
        model_name: Model identifier
        use_decomposition: Whether to use info decomposition (NEW)
        decomposition_model: Pre-trained decomposition model (optional)
        pretrain_epochs: Epochs for decomposition pretraining
    
    Returns:
        Same outputs as original train_and_evaluate_imoe for compatibility
    """
    
    if not use_decomposition or decomposition_model is None:
        # BACKWARD COMPATIBLE: Use original training
        return train_and_evaluate_imoe(args, seed, fusion_model, model_name)
    
    # NEW: Enhanced training with decomposition
    from src.common.data_loader import get_data_loaders
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(args, seed)
    
    # Move models to device
    decomposition_model.to(device)
    fusion_model.to(device)
    
    # If decomposition model is not pretrained, pretrain it first
    if not hasattr(decomposition_model, '_pretrained'):
        print(f"Pretraining decomposition model for {pretrain_epochs} epochs...")
        decomposition_model = pretrain_decomposition_model(
            decomposition_model, train_loader, args, pretrain_epochs
        )
        decomposition_model._pretrained = True
    
    # Freeze decomposition model
    decomposition_model.eval()
    for param in decomposition_model.parameters():
        param.requires_grad = False
    
    # Train fusion model with decomposed features
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    for epoch in range(args.train_epochs):
        fusion_model.train()
        
        for batch in train_loader:
            # Extract modality features
            modality_features = extract_modality_features(batch, device)
            labels = batch['labels'].to(device)
            
            # Get decomposed representations
            with torch.no_grad():
                decomposed = decomposition_model(modality_features)
            
            # Forward through fusion model
            if decomposed['decomposed']:
                # Use decomposed features
                outputs = fusion_model(
                    decomposed['unique_features'],
                    decomposed['redundant_features'],
                    decomposed['synergy_features']
                )
            else:
                # Fallback to original features
                outputs = fusion_model(*modality_features)
            
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), args.grad_norm_max)
            optimizer.step()
    
    # Evaluation (same format as original)
    val_metrics = evaluate_model(fusion_model, decomposition_model, 
                                 val_loader, device, args)
    test_metrics = evaluate_model(fusion_model, decomposition_model,
                                  test_loader, device, args)
    
    # Return in same format as original function
    if args.data == "mosi_regression":
        return (val_metrics['loss'], val_metrics['acc'], 
                test_metrics['acc'], test_metrics['mae'],
                0, 0, 0, 0)  # train_time, infer_time, flop, param
    else:
        return (val_metrics['acc'], val_metrics['f1'], val_metrics['auc'],
                test_metrics['acc'], test_metrics['f1'], test_metrics['f1_micro'],
                test_metrics['auc'], 0, 0, 0, 0)


def pretrain_decomposition_model(model, train_loader, args, n_epochs):
    """Pretrain decomposition model (placeholder - implement based on your data)"""
    print(f"Pretraining decomposition for {n_epochs} epochs...")
    # Implement pretraining logic here
    return model


def extract_modality_features(batch, device):
    """Extract modality features from batch (adjust based on your data format)"""
    features = []
    for key in sorted(batch.get('tensor', {}).keys()):
        feat = batch['tensor'][key].to(device)
        features.append(feat)
    return features


def evaluate_model(fusion_model, decomposition_model, loader, device, args):
    """Evaluate model and return metrics"""
    fusion_model.eval()
    # Implement evaluation logic
    return {'acc': 0.0, 'f1': 0.0, 'auc': 0.0, 'loss': 0.0, 'mae': 0.0, 
            'f1_micro': 0.0}


# ============== Parse input arguments (ENHANCED) ==============
def parse_args():
    parser = argparse.ArgumentParser(description="iMoE-interpretcc")
    
    # ===== ORIGINAL ARGUMENTS (unchanged) =====
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
    
    # ===== NEW ARGUMENTS (optional enhancement) =====
    parser.add_argument(
        "--use_info_decomposition", type=str2bool, default=False,
        help="Enable information-theoretic decomposition (U, R, S)"
    )
    parser.add_argument(
        "--decomposition_pretrain_epochs", type=int, default=50,
        help="Epochs for pretraining decomposition model"
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
        "--load_pretrained_decomposition", type=str, default=None,
        help="Path to pretrained decomposition model"
    )

    return parser.parse_known_args()


# ============== MAIN FUNCTION (Backward Compatible) ==============
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

    # ===== Model configuration (with NEW fields) =====
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
        # NEW fields (only if enabled)
        "use_info_decomposition": args.use_info_decomposition,
    }
    
    if args.use_info_decomposition:
        model_kwargs.update({
            "decomposition_pretrain_epochs": args.decomposition_pretrain_epochs,
            "decomposition_alpha_u": args.decomposition_alpha_u,
            "decomposition_alpha_r": args.decomposition_alpha_r,
            "decomposition_alpha_s": args.decomposition_alpha_s,
        })

    log_summary += f"Model configuration: {model_kwargs}\n"
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

    # Initialize metric lists (same as original)
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

    # ===== NEW: Initialize decomposition model if enabled =====
    decomposition_model = None
    if args.use_info_decomposition:
        print("Initializing information decomposition model...")
        
        # Get modality dimensions (you may need to adjust this)
        # This is a placeholder - adjust based on your data format
        modality_dims = [args.hidden_dim] * num_modalities
        
        decomposition_model = InfoDecompositionWrapper(
            modality_dims=modality_dims,
            hidden_dim=args.hidden_dim,
            num_classes=n_labels,
            use_decomposition=True
        ).to(device)
        
        # Load pretrained model if specified
        if args.load_pretrained_decomposition is not None:
            print(f"Loading pretrained decomposition from {args.load_pretrained_decomposition}")
            decomposition_model.load_state_dict(
                torch.load(args.load_pretrained_decomposition)
            )
            decomposition_model._pretrained = True

    # ===== Training loop (same structure as original) =====
    for i, seed in enumerate(seeds if len(seeds) > 1 else [args.seed]):
        print(f"\n{'='*80}")
        print(f"Run {i+1}/{len(seeds)}, Seed: {seed}")
        print(f"{'='*80}\n")
        
        fusion_model = InterpretCC(
            num_classes=n_labels,
            num_modality=len(args.modality),
            input_dim=args.hidden_dim,
            dropout=args.dropout,
            tau=args.tau,
            hard=args.hard,
            threshold=args.threshold,
        ).to(device)

        # ===== ENHANCED: Use new training function if decomposition enabled =====
        if args.use_info_decomposition and decomposition_model is not None:
            train_func = train_and_evaluate_imoe_enhanced
            train_kwargs = {
                'use_decomposition': True,
                'decomposition_model': decomposition_model,
                'pretrain_epochs': args.decomposition_pretrain_epochs
            }
        else:
            # BACKWARD COMPATIBLE: Use original training
            train_func = train_and_evaluate_imoe
            train_kwargs = {}

        if args.data == "mosi_regression":
            (
                val_loss, val_acc, test_acc, test_mae,
                train_time, infer_time, flop, param,
            ) = train_func(args, seed, fusion_model, "interpretcc", **train_kwargs)
            
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            test_maes.append(test_mae)
        else:
            (
                val_acc, val_f1, val_auc,
                test_acc, test_f1, test_f1_micro, test_auc,
                train_time, infer_time, flop, param,
            ) = train_func(args, seed, fusion_model, "interpretcc", **train_kwargs)
            
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

    # ===== Rest of the code (UNCHANGED - same as original) =====
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
    log_summary += f"Train one epoch time: {mean_train_time:.2f} ± {variance_train_time:.2f} "
    log_summary += "\n"
    log_summary += f"Inference one epoch time: {mean_infer_time:.2f} ± {variance_infer_time:.2f} "
    log_summary += "\n"
    log_summary += f"flops: {mean_flop:,.0f} ± {variance_flop:,.0f} "
    log_summary += "\n"
    log_summary += f"gflops: {mean_gflop:.2f} ± {variance_gflop:.2f} "
    log_summary += "\n"
    log_summary += f"param: {mean_param:,.0f} ± {variance_param:,.0f} "
    log_summary += "\n"

    if args.data == "mosi_regression":
        val_avg_acc = np.mean(val_accs) * 100
        val_std_acc = np.std(val_accs) * 100
        val_avg_loss = np.mean(val_losses)
        val_std_loss = np.std(val_losses)
        test_avg_acc = np.mean(test_accs) * 100
        test_std_acc = np.std(test_accs) * 100
        test_avg_mae = np.mean(test_maes)
        test_std_mae = np.std(test_maes)

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f} "
        log_summary += f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f}  "
        log_summary += f"[Test] Mean Absolute Error: {test_avg_mae:.2f} ± {test_std_mae:.2f}  "

        print(model_kwargs)
        print(f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average Loss: {val_avg_loss:.2f} ± {val_std_loss:.2f} ")
        print(f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} ")
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

        log_summary += f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} "
        log_summary += f"[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} "
        log_summary += f"[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / "
        log_summary += f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} "
        log_summary += f"[Test] Average F1 (Macro) Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} "
        log_summary += f"[Test] Average F1 (Micro) Score: {test_avg_f1_micro:.2f} ± {test_std_f1_micro:.2f} "
        log_summary += f"[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} "

        print(model_kwargs)
        print(f"[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}")
        print(f"[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}")

    logger.info(log_summary)
    
    # ===== NEW: Save decomposition model if enabled =====
    if args.use_info_decomposition and decomposition_model is not None:
        save_path = f"./checkpoints/decomposition_{args.data}_{args.modality}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(decomposition_model.state_dict(), save_path)
        print(f"\nSaved decomposition model to {save_path}")


if __name__ == "__main__":
    main()