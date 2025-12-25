"""
Enhanced src/imoe/imoe_train_enhanced.py

This is the CORRECT approach - enhance the existing train_and_evaluate_imoe function
with optional decomposition support, keeping the exact same signature.

Save this as: src/imoe/imoe_train_enhanced.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from copy import deepcopy
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
import time

# Import your existing data loaders (unchanged)
from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)
from src.common.datasets.MultiModalDataset import create_loaders

from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    plot_interaction_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
    visualize_expert_logits_distribution,
    set_style,
)

from src.imoe.InteractionMoE import InteractionMoE
from src.imoe.InteractionMoERegression import InteractionMoERegression

set_style()


# ============== Information Decomposition Components ==============

class ModalityDecompositionBranch(nn.Module):
    """Decomposes modality into U, R, S components"""
    
    def __init__(self, in_dim, out_dim, num_other_modalities, dropout=0.3):
        super().__init__()
        self.num_other = num_other_modalities
        
        # Unique information encoder
        self.fc_unique = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
        # Redundant information encoders
        self.fc_redundant = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim)
            ) for _ in range(num_other_modalities)
        ])
        
        # Synergistic information encoder
        self.fc_synergy = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x):
        unique = self.fc_unique(x)
        redundant = [fc(x) for fc in self.fc_redundant]
        synergy = self.fc_synergy(x)
        return unique, redundant, synergy


class InfoDecompositionPreprocessor(nn.Module):
    """
    Preprocessing module for information decomposition.
    Applied BEFORE encoders to decompose raw features.
    """
    
    def __init__(self, modality_dims, hidden_dim, num_modalities, enabled=False):
        super().__init__()
        self.enabled = enabled
        self.num_modalities = num_modalities
        
        if not enabled:
            return
        
        # Create decomposition branches for each modality
        self.branches = nn.ModuleList([
            ModalityDecompositionBranch(
                in_dim=dim,
                out_dim=hidden_dim,
                num_other_modalities=num_modalities - 1,
                dropout=0.3
            ) for dim in modality_dims
        ])
        
        # Projections to restore original dimensions
        self.restore_projections = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in modality_dims
        ])
    
    def forward(self, modality_features):
        """
        Args:
            modality_features: List of tensors, each [batch, original_dim]
            
        Returns:
            If enabled: decomposed and recombined features
            If disabled: original features (passthrough)
        """
        if not self.enabled:
            return modality_features
        
        decomposed_features = []
        
        for i, (feat, branch, restore) in enumerate(
            zip(modality_features, self.branches, self.restore_projections)
        ):
            # Decompose into U, R, S
            unique, redundant, synergy = branch(feat)
            
            # Aggregate redundant components
            if redundant:
                redundant_agg = torch.stack(redundant).mean(dim=0)
            else:
                redundant_agg = torch.zeros_like(unique)
            
            # Combine components (weighted sum)
            combined = unique + redundant_agg + synergy
            
            # Project back to original dimension
            reconstructed = restore(combined)
            decomposed_features.append(reconstructed)
        
        return decomposed_features, (unique, redundant_agg, synergy)


def compute_decomposition_loss(unique_reps, redundant_reps, synergy_reps, 
                               alpha_u=1.0, alpha_r=1.0, alpha_s=0.5):
    """
    Compute information-theoretic loss for decomposition.
    """
    loss = 0.0
    
    # 1. Uniqueness: minimize correlation with others
    for i in range(len(unique_reps)):
        for j in range(i + 1, len(unique_reps)):
            u_i = unique_reps[i].reshape(-1, unique_reps[i].shape[-1])
            u_j = unique_reps[j].reshape(-1, unique_reps[j].shape[-1])
            
            # Normalize
            u_i_norm = u_i - u_i.mean(dim=0)
            u_j_norm = u_j - u_j.mean(dim=0)
            
            # Correlation
            cov = torch.mm(u_i_norm.t(), u_j_norm) / u_i.shape[0]
            loss += alpha_u * torch.norm(cov, p='fro')
    
    # 2. Redundancy: maximize correlation between redundant parts
    if len(redundant_reps) > 1:
        for i in range(len(redundant_reps)):
            for j in range(i + 1, len(redundant_reps)):
                r_i = redundant_reps[i].reshape(-1, redundant_reps[i].shape[-1])
                r_j = redundant_reps[j].reshape(-1, redundant_reps[j].shape[-1])
                
                r_i_norm = r_i - r_i.mean(dim=0)
                r_j_norm = r_j - r_j.mean(dim=0)
                
                cov = torch.mm(r_i_norm.t(), r_j_norm) / r_i.shape[0]
                loss -= alpha_r * torch.norm(cov, p='fro')  # Negative to maximize
    
    return loss


# ============== Enhanced Training Function (SAME SIGNATURE) ==============

def train_and_evaluate_imoe(args, seed, fusion_model, fusion):
    """
    Enhanced train_and_evaluate_imoe with optional information decomposition.
    
    CRITICAL: Same signature as original - fully backward compatible.
    Decomposition is controlled by args.use_info_decomposition flag.
    
    Args:
        args: Training arguments
        seed: Random seed
        fusion_model: Fusion model (InterpretCC - UNCHANGED)
        fusion: Name of fusion method
        
    Returns:
        Same outputs as original function
    """
    
    # Check if decomposition is enabled
    use_decomposition = getattr(args, 'use_info_decomposition', False)
    decomp_alpha_u = getattr(args, 'decomposition_alpha_u', 1.0)
    decomp_alpha_r = getattr(args, 'decomposition_alpha_r', 1.0)
    decomp_alpha_s = getattr(args, 'decomposition_alpha_s', 0.5)
    decomp_weight = getattr(args, 'decomposition_loss_weight', 0.01)
    
    if use_decomposition:
        print("\n" + "="*80)
        print("Enhanced Training with Information-Theoretic Decomposition")
        print(f"Alpha U: {decomp_alpha_u}, Alpha R: {decomp_alpha_r}, Alpha S: {decomp_alpha_s}")
        print(f"Decomposition Loss Weight: {decomp_weight}")
        print("="*80 + "\n")
    
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
    num_modalities = len(args.modality)

    # ===== Data Loading (UNCHANGED) =====
    if args.data == "adni":
        (data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
         n_labels, input_dims, transforms, masks, observed_idx_arr, _, _,
        ) = load_and_preprocess_data_adni(args)
    elif args.data == "mimic":
        (data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
         n_labels, input_dims, transforms, masks, observed_idx_arr, _, _,
        ) = load_and_preprocess_data_mimic(args)
    elif args.data == "mosi":
        (data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
         n_labels, input_dims, transforms, masks, observed_idx_arr, _, _,
        ) = load_and_preprocess_data_mosi(args)
    elif args.data == "enrico":
        (data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
         n_labels, input_dims, transforms, masks, observed_idx_arr, _, _,
        ) = load_and_preprocess_data_enrico(args)
    elif args.data == "mmimdb":
        (data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
         n_labels, input_dims, transforms, masks, observed_idx_arr, _, _,
        ) = load_and_preprocess_data_mmimdb(args)
    elif args.data == "mosi_regression":
        (data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids,
         n_labels, input_dims, transforms, masks, observed_idx_arr, _, _,
        ) = load_and_preprocess_data_mosi_regression(args)

    train_loader, val_loader, test_loader = create_loaders(
        data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids,
        args.batch_size, args.num_workers, args.pin_memory, input_dims,
        transforms, masks, args.use_common_ids, dataset=args.data,
    )

    # ===== NEW: Initialize decomposition preprocessor =====
    decomp_preprocessor = None
    if use_decomposition:
        print(f"Initializing decomposition preprocessor with dims: {input_dims}")
        decomp_preprocessor = InfoDecompositionPreprocessor(
            modality_dims=input_dims,
            hidden_dim=args.hidden_dim,
            num_modalities=num_modalities,
            enabled=True
        ).to(device)

    # ===== Model Initialization (UNCHANGED) =====
    if args.data == "mosi_regression":
        ensemble_model = InteractionMoERegression(
            num_modalities=num_modalities,
            fusion_model=deepcopy(fusion_model),
            fusion_sparse=args.fusion_sparse,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
        ).to(device)
    else:
        ensemble_model = InteractionMoE(
            num_modalities=num_modalities,
            fusion_model=deepcopy(fusion_model),
            fusion_sparse=args.fusion_sparse,
            hidden_dim=args.hidden_dim,
            hidden_dim_rw=args.hidden_dim_rw,
            num_layer_rw=args.num_layer_rw,
            temperature_rw=args.temperature_rw,
        ).to(device)

    # ===== NEW: Add decomposition parameters to optimizer =====
    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]
    if use_decomposition and decomp_preprocessor is not None:
        params += list(decomp_preprocessor.parameters())
        print(f"Added {sum(p.numel() for p in decomp_preprocessor.parameters())} decomposition parameters")

    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    # ===== Loss Function (UNCHANGED) =====
    if args.data in ["adni", "enrico", "mosi"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "mosi_regression":
        criterion = torch.nn.SmoothL1Loss()
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()

    # ===== Training Setup (UNCHANGED) =====
    if args.data == "mosi_regression":
        best_val_loss = 100000
    elif args.data == "mmimdb":
        best_val_f1 = 0
    else:
        best_val_acc = 0.0

    if args.fusion_sparse:
        plotting_total_losses = {"task": [], "interaction": [], "gate": []}
    else:
        plotting_total_losses = {"task": [], "interaction": []}
    
    if use_decomposition:
        plotting_total_losses["decomposition"] = []

    plotting_interaction_losses = {}
    for i in range(len(args.modality)):
        plotting_interaction_losses[f"uni_{i+1}"] = []
    plotting_interaction_losses["syn"] = []
    plotting_interaction_losses["red"] = []

    train_time = 0

    # ===== TRAINING LOOP =====
    for epoch in trange(args.train_epochs):
        epoch_start_time = time.time()
        
        ensemble_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
        if use_decomposition and decomp_preprocessor is not None:
            decomp_preprocessor.train()

        batch_task_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []
        batch_interaction_losses = []
        if use_decomposition:
            batch_decomp_losses = []

        num_interaction_experts = len(args.modality) + 2
        interaction_loss_sums = [0] * num_interaction_experts
        minibatch_count = len(train_loader)

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            # ===== NEW: Apply decomposition preprocessing =====
            decomposed_components = None
            if use_decomposition and decomp_preprocessor is not None:
                # Extract raw features (before encoders)
                raw_features = [batch_samples[mod] for mod in sorted(batch_samples.keys())]
                
                # Apply decomposition
                processed_features, decomp_components = decomp_preprocessor(raw_features)
                
                # Update batch_samples with processed features
                for i, mod in enumerate(sorted(batch_samples.keys())):
                    batch_samples[mod] = processed_features[i]

            # ===== Encode features (UNCHANGED) =====
            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            # ===== Forward pass (UNCHANGED) =====
            if args.fusion_sparse:
                _, _, outputs, interaction_losses, gate_losses = ensemble_model(fusion_input)
            else:
                _, _, outputs, interaction_losses = ensemble_model(fusion_input)

            # ===== Task loss (UNCHANGED) =====
            if args.data == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)

            interaction_loss = sum(interaction_losses) / (len(args.modality) + 2)
            
            # ===== NEW: Decomposition loss =====
            decomp_loss = torch.tensor(0.0).to(device)
            if use_decomposition and decomp_components is not None:
                unique, redundant_agg, synergy = decomp_components
                
                # Create lists for loss computation
                unique_reps = [unique]
                redundant_reps = [redundant_agg]
                synergy_reps = [synergy]
                
                decomp_loss = compute_decomposition_loss(
                    unique_reps, redundant_reps, synergy_reps,
                    alpha_u=decomp_alpha_u, 
                    alpha_r=decomp_alpha_r, 
                    alpha_s=decomp_alpha_s
                )

            # ===== Total loss =====
            if args.fusion_sparse:
                gate_loss = torch.mean(torch.tensor(gate_losses))
                loss = (task_loss + 
                       args.interaction_loss_weight * interaction_loss + 
                       args.gate_loss_weight * gate_loss)
            else:
                loss = task_loss + args.interaction_loss_weight * interaction_loss
            
            if use_decomposition:
                loss = loss + decomp_weight * decomp_loss

            loss.backward()
            optimizer.step()

            batch_task_losses.append(task_loss.item())
            batch_interaction_losses.append(interaction_loss.item())
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())
            if use_decomposition:
                batch_decomp_losses.append(decomp_loss.item())

            for idx, loss_val in enumerate(interaction_losses):
                interaction_loss_sums[idx] += loss_val.item()

            if args.data == "enrico":
                torch.nn.utils.clip_grad_norm_(params, 1.0)

        epoch_end_time = time.time()
        train_epoch_time = epoch_end_time - epoch_start_time
        train_time += train_epoch_time

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        plotting_total_losses["interaction"].append(np.mean(batch_interaction_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))
        if use_decomposition:
            plotting_total_losses["decomposition"].append(np.mean(batch_decomp_losses))

        for i in range(len(args.modality)):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i+1}"].append(avg_loss)

        plotting_interaction_losses["syn"].append(interaction_loss_sums[-2] / minibatch_count)
        plotting_interaction_losses["red"].append(interaction_loss_sums[-1] / minibatch_count)

        # ===== VALIDATION (Enhanced to use decomposition if enabled) =====
        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()
        if use_decomposition and decomp_preprocessor is not None:
            decomp_preprocessor.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        val_losses = []

        with torch.no_grad():
            for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                batch_samples = {
                    k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
                }
                batch_labels = batch_labels.to(device, non_blocking=True)

                # Apply decomposition if enabled
                if use_decomposition and decomp_preprocessor is not None:
                    raw_features = [batch_samples[mod] for mod in sorted(batch_samples.keys())]
                    processed_features, _ = decomp_preprocessor(raw_features)
                    for i, mod in enumerate(sorted(batch_samples.keys())):
                        batch_samples[mod] = processed_features[i]

                fusion_input = []
                for i, (modality, samples) in enumerate(batch_samples.items()):
                    encoded_samples = encoder_dict[modality](samples)
                    fusion_input.append(encoded_samples)

                _, _, outputs = ensemble_model.inference(fusion_input)

                if args.data == "mosi_regression":
                    val_loss = criterion(outputs, batch_labels.unsqueeze(1))
                    val_losses.append(val_loss.item())
                    all_preds.extend(outputs.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                else:
                    if args.data == "mmimdb":
                        val_loss = criterion(outputs, batch_labels.float())
                    else:
                        val_loss = criterion(outputs, batch_labels)
                    val_losses.append(val_loss.item())
                    
                    if args.data == "mmimdb":
                        preds = torch.sigmoid(outputs).round()
                    else:
                        _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    
                    if args.data in ["mimic", "mosi"]:
                        all_probs.extend(
                            torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        )
                    else:
                        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        all_probs.extend(probs)
                        if probs.shape[1] != n_labels:
                            raise ValueError("Incorrect output shape from the model")

        # ===== Validation metrics (Enhanced with decomposition info) =====
        val_loss = np.mean(val_losses)
        
        if args.data == "mosi_regression":
            val_acc = accuracy_score((np.array(all_preds) > 0), (np.array(all_labels) > 0))
            
            decomp_str = f", Decomp: {np.mean(batch_decomp_losses):.4f}" if use_decomposition else ""
            print(f"[Seed {seed}] [Epoch {epoch+1}/{args.train_epochs}] Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}{decomp_str}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                print(f"  (**Best**) Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}")
                
                best_model_fus = deepcopy(ensemble_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                if use_decomposition and decomp_preprocessor is not None:
                    best_model_decomp = deepcopy(decomp_preprocessor.state_dict())
                
                if args.save:
                    best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                    best_model_enc_cpu = {
                        modality: {k: v.cpu() for k, v in enc_state.items()}
                        for modality, enc_state in best_model_enc.items()
                    }
        else:
            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average="macro")
            
            if args.data == "enrico":
                val_auc = roc_auc_score(np.array(all_labels), np.array(all_probs),
                                       multi_class="ovo", labels=list(range(n_labels)))
            elif args.data in ["mimic", "mosi"]:
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0
            elif args.data == "adni":
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            decomp_str = f", Decomp: {np.mean(batch_decomp_losses):.4f}" if use_decomposition else ""
            print(f"[Seed {seed}] [Epoch {epoch+1}/{args.train_epochs}] Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}{decomp_str}")

            should_save = False
            if args.data == "mmimdb":
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    should_save = True
            else:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    should_save = True
            
            if should_save:
                print(f"  (**Best**) Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                best_model_fus = deepcopy(ensemble_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                if use_decomposition and decomp_preprocessor is not None:
                    best_model_decomp = deepcopy(decomp_preprocessor.state_dict())
                
                if args.save:
                    best_model_fus_cpu = {k: v.cpu() for k, v in best_model_fus.items()}
                    best_model_enc_cpu = {
                        modality: {k: v.cpu() for k, v in enc_state.items()}
                        for modality, enc_state in best_model_enc.items()
                    }

    # ===== Continue with rest of original code (testing, visualization, etc.) =====
    # (Rest is identical to original - I'll continue in next message due to length)
    
    total_param = parameter_count(ensemble_model)[""]
    if use_decomposition and decomp_preprocessor is not None:
        total_param += parameter_count(decomp_preprocessor)[""]
    total_flop = 0

    plot_total_loss_curves(args, plotting_total_losses=plotting_total_losses,
                           framework="imoe", fusion=fusion)
    plot_interaction_loss_curves(args, plotting_interaction_losses=plotting_interaction_losses,
                                 framework="imoe", fusion=fusion)

    # Save model (enhanced to include decomposition)
    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/imoe/{fusion}/{args.data}").mkdir(exist_ok=True, parents=True)

        if args.data == "mmimdb":
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
        elif args.data == "mosi_regression":
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_loss_{best_val_loss:.2f}.pth"
        else:
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc:.2f}.pth"
        
        save_dict = {
            "ensemble_model": best_model_fus_cpu,
            "encoder_dict": best_model_enc_cpu
        }
        if use_decomposition and 'best_model_decomp' in locals():
            save_dict["decomposition_model"] = {k: v.cpu() for k, v in best_model_decomp.items()}
        
        torch.save(save_dict, save_path)
        print(f"Best model saved to {save_path}")

    # Load best model for testing (enhanced)
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()
    
    if use_decomposition and decomp_preprocessor is not None and 'best_model_decomp' in locals():
        decomp_preprocessor.load_state_dict(best_model_decomp)
        decomp_preprocessor.eval()

    # ===== TEST EVALUATION (Enhanced with decomposition) =====
    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    test_losses = []
    all_routing_weights = []
    num_experts = len(args.modality) + 2
    all_expert_outputs = [[] for _ in range(num_experts)]

    infer_time = 0

    with torch.no_grad():
        epoch_start_time = time.time()

        for (batch_samples, batch_ids, batch_labels, batch_mcs, batch_observed) in test_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)

            # Apply decomposition if enabled
            if use_decomposition and decomp_preprocessor is not None:
                raw_features = [batch_samples[mod] for mod in sorted(batch_samples.keys())]
                processed_features, _ = decomp_preprocessor(raw_features)
                for i, mod in enumerate(sorted(batch_samples.keys())):
                    batch_samples[mod] = processed_features[i]

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            expert_outputs, routing_weights, outputs = ensemble_model.inference(fusion_input)

            for expert_idx in range(num_experts):
                all_expert_outputs[expert_idx].extend(expert_outputs[expert_idx].cpu().numpy())

            all_routing_weights.extend(routing_weights.cpu().numpy())

            if args.data == "mosi_regression":
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
            else:
                if args.data == "mmimdb":
                    preds = torch.sigmoid(outputs).round()
                else:
                    _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_ids.extend(batch_ids.cpu().numpy())

                if args.data in ["mimic", "mosi"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
                else:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )

        epoch_end_time = time.time()
        infer_epoch_time = epoch_end_time - epoch_start_time
        infer_time += infer_epoch_time

    # Visualization (unchanged)
    visualize_expert_logits(expert_outputs, routing_weights, outputs, args,
                           framework="imoe", fusion=fusion)
    visualize_expert_logits_distribution(all_expert_outputs, args,
                                        framework="imoe", fusion=fusion)
    visualize_sample_weights(all_routing_weights, args, framework="imoe", fusion=fusion)

    # Final metrics and return (unchanged)
    if args.data == "mosi_regression":
        all_binary_preds = np.array(all_preds) > 0
        all_labels = np.array(all_labels) > 0
        test_acc = accuracy_score(all_binary_preds, all_labels)
        test_mae = mean_absolute_error(all_preds, all_labels)

        now = datetime.now()
        save_dir = Path(f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}")
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (best_val_loss, best_val_acc, test_acc, test_mae,
                train_time / args.train_epochs, infer_time, total_flop, total_param)
    else:
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
        test_f1_micro = f1_score(all_labels, all_preds, average="micro")
        
        if args.data == "enrico":
            test_auc = roc_auc_score(np.array(all_labels), np.array(all_probs),
                                    multi_class="ovo", labels=list(range(n_labels)))
        elif args.data in ["mimic", "mosi"]:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "mmimdb":
            test_auc = 0
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

        now = datetime.now()
        save_dir = Path(f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}")
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1,
                test_f1_micro, test_auc, train_time / args.train_epochs,
                infer_time, total_flop, total_param)