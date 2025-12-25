import torch
from tqdm import trange
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from copy import deepcopy
from datetime import datetime
from fvcore.nn import FlopCountAnalysis, parameter_count
import time

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

def train_and_evaluate_imoe_with_pretrain(args, seed, fusion_model, model_name, device):
    """
    Unified training: Pre-train decomposition → Train specialized experts
    
    Returns all metrics from both phases in a single run
    """
    import time
    from torch.optim import Adam, AdamW
    from torch.nn import functional as F
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    train_loader = get_dataloader(args, split='train', seed=seed)
    val_loader = get_dataloader(args, split='val', seed=seed)
    test_loader = get_dataloader(args, split='test', seed=seed)
    
    # Get loss function based on task
    if args.data == "mosi_regression":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # ========================================================================
    # PHASE 1: PRE-TRAIN DECOMPOSITION
    # ========================================================================
    pretrain_loss = 0.0
    pretrain_time = 0.0
    
    if args.use_pretrain and args.use_decomposition:
        print(f"\n{'='*70}")
        print(f"PHASE 1: Pre-training Decomposition ({args.pretrain_method})")
        print(f"{'='*70}")
        
        # Only optimize decomposer parameters
        decomposer_params = list(fusion_model.decomposer.parameters())
        optimizer_pretrain = Adam(decomposer_params, lr=args.pretrain_lr, 
                                 weight_decay=args.pretrain_weight_decay)
        
        start_time = time.time()
        best_pretrain_loss = float('inf')
        
        for epoch in range(args.pretrain_epochs):
            fusion_model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Extract modality features
                modality_features = extract_modality_features(batch, args, device)
                
                optimizer_pretrain.zero_grad()
                
                # Get decomposition (without task prediction)
                decomposition = fusion_model.decomposer(modality_features)
                unique = decomposition['unique']
                redundant = decomposition['redundant']
                synergistic = decomposition['synergistic']
                
                # Compute pre-training loss
                if args.pretrain_method == 'reconstruction':
                    loss = reconstruction_loss(
                        modality_features, unique, redundant, synergistic
                    )
                elif args.pretrain_method == 'contrastive':
                    loss = contrastive_loss(
                        modality_features, unique, redundant, synergistic
                    )
                elif args.pretrain_method == 'mutual_info':
                    loss = mutual_info_loss(
                        modality_features, unique, redundant, synergistic
                    )
                
                loss.backward()
                optimizer_pretrain.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{args.pretrain_epochs}: Loss={avg_loss:.4f}")
            
            if avg_loss < best_pretrain_loss:
                best_pretrain_loss = avg_loss
        
        pretrain_time = time.time() - start_time
        pretrain_loss = best_pretrain_loss
        
        print(f"✓ Pre-training completed: Loss={pretrain_loss:.4f}, Time={pretrain_time:.2f}s")
        
        # Freeze decomposer if requested
        if args.freeze_decomposer:
            for param in fusion_model.decomposer.parameters():
                param.requires_grad = False
            print("✓ Decomposer frozen for Phase 2")
    
    # ========================================================================
    # PHASE 2: TRAIN SPECIALIZED EXPERTS
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: Training Specialized Experts")
    print(f"{'='*70}")
    
    # Optimize all parameters (or just experts if decomposer is frozen)
    if args.freeze_decomposer and args.use_pretrain:
        # Only optimize non-decomposer parameters
        trainable_params = [
            p for name, p in fusion_model.named_parameters() 
            if 'decomposer' not in name and p.requires_grad
        ]
    else:
        trainable_params = fusion_model.parameters()
    
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_epochs
    )
    
    # Training tracking
    best_val_metric = 0 if args.data != "mosi_regression" else float('inf')
    best_epoch = 0
    
    # For interpretability tracking
    train_unique_ratios = []
    train_redundant_ratios = []
    train_synergistic_ratios = []
    train_expert_utils = []
    
    train_start_time = time.time()
    
    for epoch in range(args.train_epochs):
        # ============ TRAINING ============
        fusion_model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            modality_features = extract_modality_features(batch, args, device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with decomposition info
            if args.use_decomposition:
                logits, decomp_info = fusion_model(
                    modality_features, 
                    return_decomposition=True
                )
                
                # Main task loss
                task_loss = criterion(logits, labels)
                
                # Additional regularization losses
                decomp = decomp_info['decomposition']
                
                # Orthogonality loss
                ortho_loss = compute_orthogonality_loss(
                    decomp['unique'], decomp['redundant'], decomp['synergistic']
                )
                
                # Load balancing loss
                routing_weights = decomp_info['routing_weights']
                balance_loss = compute_load_balancing_loss(routing_weights)
                
                # Total loss
                loss = (task_loss + 
                       args.orthogonality_loss_weight * ortho_loss +
                       args.load_balancing_weight * balance_loss)
                
                # Track decomposition ratios
                if args.log_interactions and batch_idx % 50 == 0:
                    with torch.no_grad():
                        u_norm = decomp['unique'].norm(dim=1).mean().item()
                        r_norm = decomp['redundant'].norm(dim=1).mean().item()
                        s_norm = decomp['synergistic'].norm(dim=1).mean().item()
                        total = u_norm + r_norm + s_norm + 1e-8
                        train_unique_ratios.append(u_norm / total)
                        train_redundant_ratios.append(r_norm / total)
                        train_synergistic_ratios.append(s_norm / total)
            else:
                logits = fusion_model(modality_features)
                loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            # Compute accuracy
            if args.data != "mosi_regression":
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if args.data != "mosi_regression" else 0
        
        # ============ VALIDATION ============
        val_metrics = evaluate_model(
            fusion_model, val_loader, criterion, args, device
        )
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            if args.data == "mosi_regression":
                print(f"  Epoch {epoch+1}/{args.train_epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}")
            else:
                print(f"  Epoch {epoch+1}/{args.train_epochs}: "
                      f"Train Acc={train_acc*100:.2f}%, "
                      f"Val Acc={val_metrics['accuracy']*100:.2f}%, "
                      f"Val F1={val_metrics['f1']*100:.2f}%")
        
        # Track best model
        if args.data == "mosi_regression":
            if val_metrics['loss'] < best_val_metric:
                best_val_metric = val_metrics['loss']
                best_epoch = epoch
                # Save best model
                best_model_state = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}
        else:
            if val_metrics['accuracy'] > best_val_metric:
                best_val_metric = val_metrics['accuracy']
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in fusion_model.state_dict().items()}
    
    train_time_per_epoch = (time.time() - train_start_time) / args.train_epochs
    
    print(f"✓ Training completed: Best epoch={best_epoch+1}")
    
    # ============ TESTING WITH BEST MODEL ============
    fusion_model.load_state_dict(best_model_state)
    fusion_model.to(device)
    
    test_start_time = time.time()
    test_metrics = evaluate_model(
        fusion_model, test_loader, criterion, args, device, 
        return_decomposition=args.log_interactions
    )
    infer_time_per_epoch = time.time() - test_start_time
    
    # Compute FLOPs and params
    flops, params = compute_model_complexity(fusion_model, args)
    
    # ============ PREPARE RETURN VALUES ============
    if args.data == "mosi_regression":
        results = (
            val_metrics['loss'],           # val_loss
            val_metrics['accuracy'],       # val_acc (for regression: correlation)
            test_metrics['accuracy'],      # test_acc
            test_metrics['mae'],           # test_mae
            train_time_per_epoch,          # train_time
            infer_time_per_epoch,          # infer_time
            flops,                         # flops
            params,                        # params
            pretrain_loss,                 # pretrain_loss
            pretrain_time,                 # pretrain_time
        )
    else:
        results = (
            val_metrics['accuracy'],       # val_acc
            val_metrics['f1'],             # val_f1
            val_metrics['auc'],            # val_auc
            test_metrics['accuracy'],      # test_acc
            test_metrics['f1'],            # test_f1
            test_metrics['f1_micro'],      # test_f1_micro
            test_metrics['auc'],           # test_auc
            train_time_per_epoch,          # train_time
            infer_time_per_epoch,          # infer_time
            flops,                         # flops
            params,                        # params
            pretrain_loss,                 # pretrain_loss
            pretrain_time,                 # pretrain_time
        )
    
    # Add interpretability metrics if requested
    if args.log_interactions and train_unique_ratios:
        mean_unique = np.mean(train_unique_ratios)
        mean_redundant = np.mean(train_redundant_ratios)
        mean_synergistic = np.mean(train_synergistic_ratios)
        
        # Expert utilization from test set
        expert_utils = test_metrics.get('expert_utilization', [0] * 6)
        
        results = results + (mean_unique, mean_redundant, mean_synergistic, expert_utils)
    
    return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_modality_features(batch, args, device):
    """Extract modality features from batch"""
    # This depends on your data format
    # Example for encoded features:
    features = []
    for m in args.modality:
        feat = batch[f'modality_{m}'].to(device)
        features.append(feat)
    return features


def evaluate_model(model, dataloader, criterion, args, device, return_decomposition=False):
    """Evaluate model on given dataloader"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    expert_utilizations = []
    
    with torch.no_grad():
        for batch in dataloader:
            modality_features = extract_modality_features(batch, args, device)
            labels = batch['label'].to(device)
            
            if args.use_decomposition and return_decomposition:
                logits, decomp_info = model(modality_features, return_decomposition=True)
                
                # Track expert utilization
                routing = decomp_info['routing_weights']
                batch_utils = []
                for key in ['unique', 'redundant', 'synergistic']:
                    batch_utils.extend(routing[key].mean(0).cpu().numpy())
                expert_utilizations.append(batch_utils)
            else:
                logits = model(modality_features)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            if args.data == "mosi_regression":
                all_preds.extend(logits.squeeze().cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
    
    metrics = {'loss': total_loss / len(dataloader)}
    
    if args.data == "mosi_regression":
        # Regression metrics
        from scipy.stats import pearsonr
        correlation, _ = pearsonr(all_preds, all_labels)
        mae = mean_absolute_error(all_labels, all_preds)
        metrics['accuracy'] = correlation
        metrics['mae'] = mae
    else:
        # Classification metrics
        metrics['accuracy'] = accuracy_score(all_labels, all_preds)
        metrics['f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        
        # AUC (handle multi-class)
        try:
            if len(np.unique(all_labels)) == 2:
                metrics['auc'] = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
            else:
                metrics['auc'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except:
            metrics['auc'] = 0.0
    
    if expert_utilizations:
        metrics['expert_utilization'] = np.mean(expert_utilizations, axis=0)
    
    return metrics


def reconstruction_loss(modality_features, unique, redundant, synergistic):
    """Reconstruction pre-training loss"""
    # Average all modality features as target
    target = torch.stack(modality_features, dim=1).mean(dim=1)
    
    # Reconstruct from decomposition
    reconstructed = unique + redundant + synergistic
    
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructed, target)
    
    # Orthogonality constraint
    ortho_loss = compute_orthogonality_loss(unique, redundant, synergistic)
    
    return recon_loss + 0.1 * ortho_loss


def contrastive_loss(modality_features, unique, redundant, synergistic, temperature=0.07):
    """Contrastive pre-training loss"""
    batch_size = unique.size(0)
    
    # Normalize
    u_norm = F.normalize(unique, dim=-1)
    r_norm = F.normalize(redundant, dim=-1)
    s_norm = F.normalize(synergistic, dim=-1)
    
    # Redundant should be similar within batch
    r_sim = torch.mm(r_norm, r_norm.t()) / temperature
    r_labels = torch.arange(batch_size).to(redundant.device)
    r_loss = F.cross_entropy(r_sim, r_labels)
    
    # Unique should be dissimilar
    u_sim = torch.mm(u_norm, u_norm.t()) / temperature
    u_loss = -torch.log(1 - F.softmax(u_sim, dim=1) + 1e-8).mean()
    
    # Synergistic should be orthogonal to U and R
    s_u_sim = F.cosine_similarity(synergistic, unique, dim=-1).abs().mean()
    s_r_sim = F.cosine_similarity(synergistic, redundant, dim=-1).abs().mean()
    s_loss = s_u_sim + s_r_sim
    
    return r_loss + u_loss + s_loss


def mutual_info_loss(modality_features, unique, redundant, synergistic):
    """Mutual information pre-training loss (simplified)"""
    # Stack modalities
    stacked = torch.stack(modality_features, dim=1)
    mean_modality = stacked.mean(dim=1)
    
    # Simplified MI estimation using correlation
    u_mi = F.cosine_similarity(unique, modality_features[0], dim=-1).mean()
    r_mi = F.cosine_similarity(redundant, mean_modality, dim=-1).mean()
    s_mi = F.cosine_similarity(synergistic, torch.cat(modality_features, dim=-1), dim=-1).mean()
    
    # Maximize MI (negate for minimization)
    return -(u_mi + r_mi + s_mi)


def compute_orthogonality_loss(unique, redundant, synergistic):
    """Encourage orthogonality between U, R, S"""
    ur = F.cosine_similarity(unique, redundant, dim=-1).abs().mean()
    us = F.cosine_similarity(unique, synergistic, dim=-1).abs().mean()
    rs = F.cosine_similarity(redundant, synergistic, dim=-1).abs().mean()
    return (ur + us + rs) / 3


def compute_load_balancing_loss(routing_weights):
    """Encourage balanced expert usage"""
    losses = []
    for key, weights in routing_weights.items():
        mean_usage = weights.mean(0)
        target = torch.ones_like(mean_usage) / len(mean_usage)
        losses.append(F.mse_loss(mean_usage, target))
    return sum(losses) / len(losses)


def compute_model_complexity(model, args):
    """Compute FLOPs and parameters"""
    from thop import profile
    
    # Create dummy input
    dummy_features = [
        torch.randn(1, args.hidden_dim).to(next(model.parameters()).device)
        for _ in range(len(args.modality))
    ]
    
    flops, params = profile(model, inputs=(dummy_features,), verbose=False)
    return flops, params
def train_and_evaluate_imoe(args, seed, fusion_model, fusion):
    """Train and evaluate interaction MoE.

    Args:
        args (argparser.args): argument
        seed (int): random seed
        ensemble_model (nn.Module): ensemble model
        fusion (str): name of fusion method

    Raises:
        ValueError

    Returns:
        tuple: (best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc)
    """
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)
    num_modalities = len(args.modality)

    if args.data == "adni":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_adni(args)
    elif args.data == "mimic":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mimic(args)
    elif args.data == "mosi":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mosi(args)
    elif args.data == "sarcasm":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_sarcasm(args)
    elif args.data == "humor":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_humor(args)
    elif args.data == "enrico":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_enrico(args)
    elif args.data == "mmimdb":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mmimdb(args)
    elif args.data == "mosi_regression":
        (
            data_dict,
            encoder_dict,
            labels,
            train_ids,
            valid_ids,
            test_ids,
            n_labels,
            input_dims,
            transforms,
            masks,
            observed_idx_arr,
            _,
            _,
        ) = load_and_preprocess_data_mosi_regression(args)

    train_loader, val_loader, test_loader = create_loaders(
        data_dict,
        observed_idx_arr,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        args.batch_size,
        args.num_workers,
        args.pin_memory,
        input_dims,
        transforms,
        masks,
        args.use_common_ids,
        dataset=args.data,
    )

    ensemble_model = InteractionMoE(
        num_modalities=num_modalities,
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=args.fusion_sparse,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

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

    params = list(ensemble_model.parameters()) + [
        param for encoder in encoder_dict.values() for param in encoder.parameters()
    ]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.data in ["adni", "enrico", "mosi", "sarcasm", "humor"]:
        criterion = torch.nn.CrossEntropyLoss()
    elif args.data == "mimic":
        criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    elif args.data == "mosi_regression":
        criterion = torch.nn.SmoothL1Loss()  # Regression
    elif args.data == "mmimdb":
        criterion = torch.nn.BCEWithLogitsLoss()

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

    plotting_interaction_losses = {}
    for i in range(len(args.modality)):
        plotting_interaction_losses[f"uni_{i+1}"] = []
    plotting_interaction_losses[f"syn"] = []
    plotting_interaction_losses[f"red"] = []

    ############ efficiency
    train_time = 0
    ############ efficiency

    for epoch in trange(args.train_epochs):
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        ensemble_model.train()

        for encoder in encoder_dict.values():
            encoder.train()

        batch_task_losses = []
        if args.fusion_sparse:
            batch_gate_losses = []
        batch_interaction_losses = []

        num_interaction_experts = len(args.modality) + 2
        interaction_loss_sums = [0] * (num_interaction_experts)
        minibatch_count = len(train_loader)

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            if args.fusion_sparse:
                _, _, outputs, interaction_losses, gate_losses = ensemble_model(
                    fusion_input
                )
            else:
                _, _, outputs, interaction_losses = ensemble_model(fusion_input)

            if args.data == "mosi_regression":
                task_loss = criterion(outputs, batch_labels.unsqueeze(1))
            else:
                task_loss = criterion(outputs, batch_labels)

            interaction_loss = sum(interaction_losses) / (len(args.modality) + 2)
            if args.fusion_sparse:
                gate_loss = torch.mean(torch.tensor(gate_losses))
                loss = (
                    task_loss
                    + args.interaction_loss_weight * interaction_loss
                    + args.gate_loss_weight * gate_loss
                )
            else:
                loss = task_loss + args.interaction_loss_weight * interaction_loss

            loss.backward()
            optimizer.step()

            batch_task_losses.append(task_loss.item())
            batch_interaction_losses.append(interaction_loss.item())
            if args.fusion_sparse:
                batch_gate_losses.append(gate_loss.item())

            for idx, loss in enumerate(interaction_losses):
                interaction_loss_sums[idx] += loss.item()

            if args.data == "enrico":
                torch.nn.utils.clip_grad_norm_(params, 1.0)

        ############ efficiency
        epoch_end_time = time.time()
        train_epoch_time = epoch_end_time - epoch_start_time
        train_time += train_epoch_time
        ############ efficiency

        plotting_total_losses["task"].append(np.mean(batch_task_losses))
        plotting_total_losses["interaction"].append(np.mean(batch_interaction_losses))
        if args.fusion_sparse:
            plotting_total_losses["gate"].append(np.mean(batch_gate_losses))

        for i in range(len(args.modality)):
            avg_loss = interaction_loss_sums[i] / minibatch_count
            plotting_interaction_losses[f"uni_{i+1}"].append(avg_loss)

        # For syn and red interaction losses
        plotting_interaction_losses["syn"].append(
            interaction_loss_sums[-2] / minibatch_count
        )
        plotting_interaction_losses["red"].append(
            interaction_loss_sums[-1] / minibatch_count
        )

        ensemble_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

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
                batch_mcs = batch_mcs.to(device, non_blocking=True)
                batch_observed = batch_observed.to(device, non_blocking=True)
                optimizer.zero_grad()

                fusion_input = []
                for i, (modality, samples) in enumerate(batch_samples.items()):
                    encoded_samples = encoder_dict[modality](samples)
                    fusion_input.append(encoded_samples)

                _, _, outputs = ensemble_model.inference(fusion_input)

                if args.data == "mosi_regression":
                    # if False:
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
                    if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                        all_probs.extend(
                            torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                            .cpu()
                            .numpy()
                        )
                    else:
                        probs = (
                            torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        )
                        all_probs.extend(probs)
                        if (
                            probs.shape[1] != n_labels
                        ):  # n_labels is the number of classes
                            raise ValueError("Incorrect output shape from the model")
        if args.data == "mosi_regression":
            val_loss = np.mean(val_losses)
            val_acc = accuracy_score(
                (np.array(all_preds) > 0), (np.array(all_labels) > 0)
            )
            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(val_losses):.2f} / Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc

                print(
                    f"[(**Best**) [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}"
                )

                best_model_fus = deepcopy(ensemble_model.state_dict())
                best_model_enc = {
                    modality: deepcopy(encoder.state_dict())
                    for modality, encoder in encoder_dict.items()
                }
                # Move the models to CPU for saving (only state_dict)
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
                val_auc = roc_auc_score(
                    np.array(all_labels),
                    np.array(all_probs),
                    multi_class="ovo",
                    labels=list(range(n_labels)),
                )
            elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                val_auc = roc_auc_score(all_labels, all_probs)
            elif args.data == "mmimdb":
                val_auc = 0
            elif args.data == "adni":
                val_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

            print(
                f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}]  Val Loss: {val_loss:.2f}, Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
            )

            if args.data == "mmimdb":
                # if False:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_val_acc = val_acc
                    best_val_auc = val_auc
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )

                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }

                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
            else:
                if val_acc > best_val_acc:
                    print(
                        f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}"
                    )
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    best_val_auc = val_auc
                    best_model_fus = deepcopy(ensemble_model.state_dict())
                    best_model_enc = {
                        modality: deepcopy(encoder.state_dict())
                        for modality, encoder in encoder_dict.items()
                    }
                    # Move the models to CPU for saving (only state_dict)
                    if args.save:
                        best_model_fus_cpu = {
                            k: v.cpu() for k, v in best_model_fus.items()
                        }
                        best_model_enc_cpu = {
                            modality: {k: v.cpu() for k, v in enc_state.items()}
                            for modality, enc_state in best_model_enc.items()
                        }
    ############ efficiency
    total_param = parameter_count(ensemble_model)[""]
    # flop = FlopCountAnalysis(ensemble_model, fusion_input)
    total_flop = 0
    ############ efficiency

    plot_total_loss_curves(
        args,
        plotting_total_losses=plotting_total_losses,
        framework="imoe",
        fusion=fusion,
    )

    plot_interaction_loss_curves(
        args,
        plotting_interaction_losses=plotting_interaction_losses,
        framework="imoe",
        fusion=fusion,
    )
    # Save the best model
    if args.save:
        Path("./saves").mkdir(exist_ok=True, parents=True)
        Path(f"./saves/imoe/{fusion}/{args.data}").mkdir(exist_ok=True, parents=True)

        if args.data == "mmimdb":
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_f1_{best_val_f1:.2f}.pth"
        elif args.data == "mosi_regression":
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_loss_{best_val_loss:.2f}.pth"
        else:
            save_path = f"./saves/imoe/{fusion}/{args.data}/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc:.2f}.pth"
        torch.save(
            {"ensemble_model": best_model_fus_cpu, "encoder_dict": best_model_enc_cpu},
            save_path,
        )

        print(f"Best model saved to {save_path}")

    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()

    ensemble_model.load_state_dict(best_model_fus)
    ensemble_model.eval()

    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []
    test_losses = []
    all_routing_weights = []
    num_experts = len(args.modality) + 2
    all_expert_outputs = [[] for _ in range(num_experts)]

    ############ efficiency
    infer_time = 0
    ############ efficiency

    with torch.no_grad():
        ############ efficiency
        epoch_start_time = time.time()
        ############ efficiency

        for (
            batch_samples,
            batch_ids,
            batch_labels,
            batch_mcs,
            batch_observed,
        ) in test_loader:
            batch_samples = {
                k: v.to(device, non_blocking=True) for k, v in batch_samples.items()
            }
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()

            fusion_input = []
            for i, (modality, samples) in enumerate(batch_samples.items()):
                encoded_samples = encoder_dict[modality](samples)
                fusion_input.append(encoded_samples)

            expert_outputs, routing_weights, outputs = ensemble_model.inference(
                fusion_input
            )

            for expert_idx in range(num_experts):
                all_expert_outputs[expert_idx].extend(
                    expert_outputs[expert_idx].cpu().numpy()
                )

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

                if args.data in ["mimic", "mosi", "sarcasm", "humor"]:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    )
                else:
                    all_probs.extend(
                        torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                    )

    ############ efficiency
    epoch_end_time = time.time()
    infer_epoch_time = epoch_end_time - epoch_start_time
    infer_time += infer_epoch_time
    ############ efficiency

    visualize_expert_logits(
        expert_outputs, routing_weights, outputs, args, framework="imoe", fusion=fusion
    )

    visualize_expert_logits_distribution(
        all_expert_outputs, args, framework="imoe", fusion=fusion
    )

    visualize_sample_weights(all_routing_weights, args, framework="imoe", fusion=fusion)

    if args.data == "mosi_regression":
        all_binary_preds = np.array(all_preds) > 0
        all_labels = np.array(all_labels) > 0
        test_acc = accuracy_score(all_binary_preds, all_labels)
        test_mae = mean_absolute_error(all_preds, all_labels)

        now = datetime.now()
        save_dir = Path(
            f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (
            best_val_loss,
            best_val_acc,
            test_acc,
            test_mae,
            train_time / args.train_epochs,
            infer_time,
            total_flop,
            total_param,
        )
    else:
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")
        test_f1_micro = f1_score(all_labels, all_preds, average="micro")
        if args.data == "enrico":
            test_auc = roc_auc_score(
                np.array(all_labels),
                np.array(all_probs),
                multi_class="ovo",
                labels=list(range(n_labels)),
            )
        elif args.data in ["mimic", "mosi", "sarcasm", "humor"]:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "mmimdb":
            test_auc = 0
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

        now = datetime.now()
        save_dir = Path(
            f"./outputs/imoe/{fusion}/{args.data}_{now.strftime('%Y-%m-%d_%H:%M:%S')}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        np.save(save_dir / "all_expert_outputs.npy", np.array(all_expert_outputs))
        np.save(save_dir / "all_routing_weights.npy", np.array(all_routing_weights))
        np.save(save_dir / "all_preds.npy", np.array(all_preds))
        np.save(save_dir / "all_labels.npy", np.array(all_labels))
        np.save(save_dir / "all_ids.npy", np.array(all_ids))

        return (
            best_val_acc,
            best_val_f1,
            best_val_auc,
            test_acc,
            test_f1,
            test_f1_micro,
            test_auc,
            train_time / args.train_epochs,
            infer_time,
            total_flop,
            total_param,
        )
