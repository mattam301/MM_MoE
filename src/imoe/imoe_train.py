# src/imoe/imoe_train.py
# ============================================================================
# Unified iMoE Training with Enhanced PID (FIXED - Matches Original Structure)
# ============================================================================
# - Does NOT modify InteractionMoE / InteractionMoERegression
# - Fixes batch collapse via per-sample adapter
# - Enhanced PID is OPTIONAL and non-breaking
# - Fully compatible with existing scripts, logging, plots, saves
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import time
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from fvcore.nn import parameter_count
from typing import List, Tuple, Optional, Dict, Any

# ===================== DATA =====================
from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)
from src.common.datasets.MultiModalDataset import create_loaders

# ===================== UTILS =====================
from src.common.utils import (
    seed_everything,
    plot_total_loss_curves,
    plot_interaction_loss_curves,
    visualize_sample_weights,
    visualize_expert_logits,
    visualize_expert_logits_distribution,
    set_style,
)

# ===================== MODELS =====================
from src.imoe.InteractionMoE import InteractionMoE
from src.imoe.InteractionMoERegression import InteractionMoERegression
# ===================== PID INSIGHTS =====================
from src.imoe.pid_insight import log_per_class_split, log_confidence_correlation, print_trend_ascii

set_style()


# ============================================================================
# SHAPE SAFETY
# ============================================================================

def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    if x.dim() > 2:
        return x.mean(dim=tuple(range(1, x.dim() - 1)))
    raise RuntimeError(f"Invalid tensor shape: {x.shape}")


# ============================================================================
# SAFE MoE ADAPTER (THE CORE FIX)
# ============================================================================

def moe_forward_per_sample(model, fusion_input):
    """
    Calls InteractionMoE one sample at a time.
    Guarantees correct [B, C] outputs and valid interaction losses.
    """
    B = fusion_input[0].shape[0]

    outputs = []
    interaction_losses_sum = None

    for b in range(B):
        single = [x[b:b+1] for x in fusion_input]
        _, _, y, inter_losses = model(single)

        outputs.append(y)

        if interaction_losses_sum is None:
            interaction_losses_sum = inter_losses
        else:
            interaction_losses_sum = [
                a + b for a, b in zip(interaction_losses_sum, inter_losses)
            ]

    outputs = torch.cat(outputs, dim=0)
    interaction_losses = [l / B for l in interaction_losses_sum]

    return outputs, interaction_losses


# ============================================================================
# SIMPLE INFORMATION DECOMPOSITION (ORIGINAL - WORKING)
# ============================================================================

class ModalityDecompositionBranch(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_other):
        super().__init__()
        self.unique = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.synergy = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.redundant = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_other)
        ])

    def forward(self, x):
        u = self.unique(x)
        s = self.synergy(x)
        r = [m(x) for m in self.redundant]
        r = torch.stack(r).mean(dim=0) if r else torch.zeros_like(u)
        return u, r, s


class InfoDecompositionPreprocessor(nn.Module):
    def __init__(self, modality_dims, hidden_dim):
        super().__init__()
        self.branches = nn.ModuleList([
            ModalityDecompositionBranch(d, hidden_dim, len(modality_dims) - 1)
            for d in modality_dims
        ])
        self.restore = nn.ModuleList([
            nn.Linear(hidden_dim, d) for d in modality_dims
        ])

    def forward(self, features):
        processed = []
        u_all, r_all, s_all = [], [], []

        for x, b, r in zip(features, self.branches, self.restore):
            u, red, syn = b(x)
            u_all.append(u)
            r_all.append(red)
            s_all.append(syn)
            processed.append(r(u + red + syn))

        return processed, (
            torch.stack(u_all).mean(0),
            torch.stack(r_all).mean(0),
            torch.stack(s_all).mean(0),
        )


def decomposition_loss(u, r, s):
    return torch.norm(u.T @ r, p="fro") + torch.norm(u.T @ s, p="fro")


# ============================================================================
# ENHANCED PID COMPONENTS (OPTIONAL ADD-ONS)
# ============================================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class VariationalBottleneck(nn.Module):
    """Optional variational bottleneck for information control."""
    
    def __init__(self, in_dim: int, latent_dim: int, beta: float = 0.001):
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
        self.beta = beta

    def forward(self, x):
        mu = self.mu(x)
        logvar = torch.clamp(self.logvar(x), -10, 2)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std) * 0.1
            z = mu + eps * std
        else:
            z = mu
        
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return z, self.beta * kl


class CrossModalAttention(nn.Module):
    """Cross-modal attention for better redundancy/synergy detection."""
    
    def __init__(self, dim: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, context):
        q = query.unsqueeze(1)
        c = context.unsqueeze(1)
        out, _ = self.attention(q, c, c)
        return self.norm(query + out.squeeze(1))


class EnhancedModalityBranch(nn.Module):
    """Enhanced decomposition branch with attention and bottleneck."""
    
    def __init__(self, in_dim, hidden_dim, num_other, use_bottleneck=True, use_attention=True):
        super().__init__()
        self.use_bottleneck = use_bottleneck
        self.use_attention = use_attention
        
        # Core encoders
        self.unique = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.synergy = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.redundant = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(max(1, num_other))
        ])
        
        # Optional enhancements
        if use_bottleneck:
            self.unique_bottleneck = VariationalBottleneck(hidden_dim, hidden_dim)
            self.synergy_bottleneck = VariationalBottleneck(hidden_dim, hidden_dim)
        
        if use_attention and num_other > 0:
            self.cross_attention = CrossModalAttention(hidden_dim)

    def forward(self, x, other_features=None):
        u = self.unique(x)
        s = self.synergy(x)
        r = [m(x) for m in self.redundant]
        r = torch.stack(r).mean(dim=0) if r else torch.zeros_like(u)
        
        kl_loss = torch.tensor(0.0, device=x.device)
        
        # Apply bottleneck if enabled
        if self.use_bottleneck:
            u, kl_u = self.unique_bottleneck(u)
            s, kl_s = self.synergy_bottleneck(s)
            kl_loss = kl_u + kl_s
        
        # Apply cross-attention if enabled and other features available
        if self.use_attention and other_features is not None and len(other_features) > 0:
            other_mean = torch.stack(other_features).mean(0)
            r = self.cross_attention(r, other_mean)
        
        return u, r, s, kl_loss


class EnhancedInfoDecomposition(nn.Module):
    """Enhanced PID with optional advanced features."""
    
    def __init__(self, modality_dims, hidden_dim, 
                 use_bottleneck=True, use_attention=True, use_aux_loss=False, num_classes=2):
        super().__init__()
        self.num_modalities = len(modality_dims)
        self.use_aux_loss = use_aux_loss
        
        self.branches = nn.ModuleList([
            EnhancedModalityBranch(d, hidden_dim, len(modality_dims) - 1, 
                                   use_bottleneck, use_attention)
            for d in modality_dims
        ])
        self.restore = nn.ModuleList([
            nn.Linear(hidden_dim, d) for d in modality_dims
        ])
        
        # Optional auxiliary predictor
        if use_aux_loss and num_classes > 1:
            self.aux_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            self.aux_predictor = None
        
        # Learnable component weights
        self.component_weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, features, labels=None):
        processed = []
        u_all, r_all, s_all = [], [], []
        total_kl = torch.tensor(0.0, device=features[0].device)
        
        # First pass: encode all
        encoded = []
        for x, branch in zip(features, self.branches):
            encoded.append(branch.unique(x))
        
        # Second pass: with cross-modal info
        for i, (x, branch, restore) in enumerate(zip(features, self.branches, self.restore)):
            other_features = [encoded[j] for j in range(len(encoded)) if j != i]
            u, r, s, kl = branch(x, other_features)
            
            u_all.append(u)
            r_all.append(r)
            s_all.append(s)
            total_kl = total_kl + kl
            
            # Weighted combination for restoration
            weights = F.softmax(self.component_weights, dim=0)
            combined = weights[0] * u + weights[1] * r + weights[2] * s
            processed.append(restore(combined))
        
        # Stack components
        u_mean = torch.stack(u_all).mean(0)
        r_mean = torch.stack(r_all).mean(0)
        s_mean = torch.stack(s_all).mean(0)
        
        # Compute losses
        losses = {"kl": total_kl / self.num_modalities}
        
        # Auxiliary prediction loss
        if self.aux_predictor is not None and labels is not None:
            aux_logits = self.aux_predictor(s_mean)
            losses["aux"] = F.cross_entropy(aux_logits, labels)
        
        return processed, (u_mean, r_mean, s_mean), losses

    def get_component_weights(self):
        return F.softmax(self.component_weights, dim=0).detach().cpu().numpy()


def enhanced_decomposition_loss(u, r, s, losses_dict, config=None):
    """Compute total decomposition loss with optional enhanced components."""
    config = config or {}
    
    # Basic orthogonality loss (always applied)
    ortho_weight = config.get("ortho_weight", 1.0)
    loss = ortho_weight * (torch.norm(u.T @ r, p="fro") + torch.norm(u.T @ s, p="fro"))
    
    # KL loss from bottleneck
    if "kl" in losses_dict:
        kl_weight = config.get("kl_weight", 0.001)
        loss = loss + kl_weight * losses_dict["kl"]
    
    # Auxiliary prediction loss
    if "aux" in losses_dict:
        aux_weight = config.get("aux_weight", 0.3)
        loss = loss + aux_weight * losses_dict["aux"]
    
    return loss


# ============================================================================
# MAIN TRAINING FUNCTION (DROP-IN REPLACEMENT)
# ============================================================================

def train_and_evaluate_imoe(args, seed, fusion_model, fusion):
    """
    SAFE, DROP-IN replacement for original train_and_evaluate_imoe.
    Preserves full API contract and fixes InteractionMoE batch collapse.
    Enhanced PID is optional and backward compatible.
    """

    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)

    # ======================================================
    # DATA LOADING (unchanged from original)
    # ======================================================
    if args.data == "mosi":
        (
            data_dict, encoder_dict, labels,
            train_ids, val_ids, test_ids,
            n_labels, input_dims, transforms,
            masks, observed_idx_arr, _, _
        ) = load_and_preprocess_data_mosi(args)
    elif args.data == "mimic":
        (
            data_dict, encoder_dict, labels,
            train_ids, val_ids, test_ids,
            n_labels, input_dims, transforms,
            masks, observed_idx_arr, _, _
        ) = load_and_preprocess_data_mimic(args)
    elif args.data == "adni":
        (
            data_dict, encoder_dict, labels,
            train_ids, val_ids, test_ids,
            n_labels, input_dims, transforms,
            masks, observed_idx_arr, _, _
        ) = load_and_preprocess_data_adni(args)
    elif args.data == "enrico":
        (
            data_dict, encoder_dict, labels,
            train_ids, val_ids, test_ids,
            n_labels, input_dims, transforms,
            masks, observed_idx_arr, _, _
        ) = load_and_preprocess_data_enrico(args)
    elif args.data == "mmimdb":
        (
            data_dict, encoder_dict, labels,
            train_ids, val_ids, test_ids,
            n_labels, input_dims, transforms,
            masks, observed_idx_arr, _, _
        ) = load_and_preprocess_data_mmimdb(args)
    else:
        raise NotImplementedError(f"Dataset {args.data} not supported")

    train_loader, val_loader, test_loader = create_loaders(
        data_dict, observed_idx_arr, labels,
        train_ids, val_ids, test_ids,
        args.batch_size, args.num_workers,
        args.pin_memory, input_dims,
        transforms, masks, args.use_common_ids,
        dataset=args.data
    )

    # ======================================================
    # MODEL
    # ======================================================
    model = InteractionMoE(
        num_modalities=len(args.modality),
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=False,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    # ======================================================
    from src.imoe.pid_insight import SimpleComponentTracker
    tracker = SimpleComponentTracker(save_path=f"./results/{args.data}_pid.csv")

    # ======================================================
    # DECOMPOSITION MODULE (OPTIONAL)
    # ======================================================
    decomp = None
    use_simple = getattr(args, 'use_info_decomposition', False)
    use_enhanced = getattr(args, 'use_enhanced_pid', False)
    print(f"Using decomposition: simple={use_simple}, enhanced={use_enhanced}")
    if use_enhanced:
        dims = [input_dims[m] for m in sorted(input_dims)]
        decomp = EnhancedInfoDecomposition(
            dims, args.hidden_dim,
            use_bottleneck=getattr(args, 'use_bottleneck', True),
            use_attention=getattr(args, 'use_attention', True),
            use_aux_loss=getattr(args, 'use_aux_predictor', False),
            num_classes=n_labels
        ).to(device)
        print("Using Enhanced PID Module")
    elif use_simple:
        dims = [input_dims[m] for m in sorted(input_dims)]
        decomp = InfoDecompositionPreprocessor(dims, args.hidden_dim).to(device)
        print("Using Simple PID Module")

    # ======================================================
    # OPTIMIZER
    # ======================================================
    params = list(model.parameters())
    for enc in encoder_dict.values():
        params += list(enc.parameters())
    if decomp:
        params += list(decomp.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ======================================================
    # TRACKING
    # ======================================================
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_state = None

    train_time = 0.0
    infer_time = 0.0

    # Decomposition config
    decomp_config = {
        "ortho_weight": getattr(args, 'decomposition_loss_weight', 1.0),
        "kl_weight": getattr(args, 'pid_kl_weight', 0.001),
        "aux_weight": getattr(args, 'pid_aux_weight', 0.3),
    }
    print(decomp)

    # ======================================================
    # TRAINING LOOP (same structure as original)
    # ======================================================
    for epoch in trange(args.train_epochs):
        start = time.time()
        model.train()
        for enc in encoder_dict.values():
            enc.train()
        if decomp:
            decomp.train()

        # ORIGINAL BATCH UNPACKING - DO NOT CHANGE
        for samples, batch_labels, *_ in train_loader:
            batch_labels = batch_labels.to(device)
            samples = {k: v.to(device) for k, v in samples.items()}
            optimizer.zero_grad()

            fusion_input = [
                ensure_2d(encoder_dict[m](samples[m]))
                for m in samples
            ]

            outputs, inter_losses = moe_forward_per_sample(model, fusion_input)
            loss = criterion(outputs, batch_labels)

            # Apply decomposition loss if enabled
            if decomp is not None:
                raw = [ensure_2d(samples[m]) for m in sorted(samples)]
                
                if isinstance(decomp, EnhancedInfoDecomposition):
                    _, (u, r, s), losses_dict = decomp(raw, batch_labels)
                    loss = loss + enhanced_decomposition_loss(u, r, s, losses_dict, decomp_config)
                else:
                    _, (u, r, s) = decomp(raw)
                    loss = loss + decomp_config["ortho_weight"] * decomposition_loss(u, r, s)
                
                # Log per-class info split every 10 epochs
                if decomp is not None and epoch % 5 == 0:  # Every 5 epochs
                    with torch.no_grad():
                        # Get validation batch for analysis
                        sample_batch = next(iter(val_loader))
                        samples, batch_labels = sample_batch[0], sample_batch[1]
                        samples = {k: v.to(device) for k, v in samples.items()}
                        batch_labels = batch_labels.to(device)
                        
                        # Encode
                        fusion_input = [ensure_2d(encoder_dict[m](samples[m])) for m in samples]
                        outputs, _ = moe_forward_per_sample(model, fusion_input)
                        preds = outputs.argmax(dim=1)
                        
                        # Get PID components
                        raw = [ensure_2d(samples[m]) for m in sorted(samples)]
                        if isinstance(decomp, EnhancedInfoDecomposition):
                            _, (u, r, s), _ = decomp(raw)
                        else:
                            _, (u, r, s) = decomp(raw)
                        
                        # === LOGGING ===
                        print(f"\n  [Epoch {epoch+1}] PID Analysis:")
                        
                        # 1. Overall split
                        u_c = u.norm(dim=1).mean().item()
                        r_c = r.norm(dim=1).mean().item()
                        s_c = s.norm(dim=1).mean().item()
                        total = u_c + r_c + s_c + 1e-8
                        print(f"    Info Split: U={u_c/total*100:.1f}% | R={r_c/total*100:.1f}% | S={s_c/total*100:.1f}%")
                        
                        # 2. Dominant component distribution
                        from src.imoe.pid_insight import log_dominant_component_distribution
                        log_dominant_component_distribution(u, r, s)
                        
                        # 3. Accuracy by component
                        from src.imoe.pid_insight import log_accuracy_by_dominant_component
                        log_accuracy_by_dominant_component(u, r, s, preds, batch_labels)
                        
                        # 4. Confidence correlation
                        log_confidence_correlation(u, r, s, outputs)
                        
                        # 5. Track for later
                        tracker.log(epoch, u, r, s, val_acc=val_acc)

            loss.backward()
            optimizer.step()

        train_time += time.time() - start
        if decomp is not None:
            tracker.save()
            tracker.print_summary()
            print_trend_ascii(tracker.data)
        # ==================================================
        # VALIDATION (same structure as original)
        # ==================================================
        model.eval()
        preds, gts, probs = [], [], []

        with torch.no_grad():
            for samples, batch_labels, *_ in val_loader:
                batch_labels = batch_labels.to(device)
                samples = {k: v.to(device) for k, v in samples.items()}

                fusion_input = [
                    ensure_2d(encoder_dict[m](samples[m]))
                    for m in samples
                ]

                outputs, _ = moe_forward_per_sample(model, fusion_input)

                p = torch.softmax(outputs, dim=1)
                preds.extend(outputs.argmax(1).cpu().numpy())
                probs.extend(p[:, 1].cpu().numpy() if n_labels == 2 else p.cpu().numpy())
                gts.extend(batch_labels.cpu().numpy())

        val_acc = accuracy_score(gts, preds)
        val_f1 = f1_score(gts, preds, average="macro")
        
        try:
            if n_labels == 2:
                val_auc = roc_auc_score(gts, probs)
            else:
                val_auc = roc_auc_score(gts, probs, multi_class="ovr")
        except:
            val_auc = 0.0

        print(
            f"[Epoch {epoch+1}/{args.train_epochs}] "
            f"Val Acc: {val_acc*100:.2f} | "
            f"Val F1: {val_f1*100:.2f} | "
            f"Val AUC: {val_auc*100:.2f}"
        )

        # Log component weights if using enhanced PID
        if isinstance(decomp, EnhancedInfoDecomposition) and epoch % 5 == 0:
            weights = decomp.get_component_weights()
            print(f"    Component weights - U: {weights[0]:.3f}, R: {weights[1]:.3f}, S: {weights[2]:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_state = deepcopy(model.state_dict())

    # ======================================================
    # LOAD BEST MODEL
    # ======================================================
    model.load_state_dict(best_state)
    model.eval()

    # ======================================================
    # TEST EVALUATION (same structure as original)
    # ======================================================
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # MOSI test loader structure
            batch_samples, batch_ids, batch_labels, batch_mcs, batch_observed = batch

            batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device)

            fusion_input = [
                ensure_2d(encoder_dict[m](batch_samples[m]))
                for m in batch_samples
            ]

            outputs, _ = moe_forward_per_sample(model, fusion_input)

            p = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(p[:, 1].cpu().numpy() if n_labels == 2 else p.cpu().numpy())

    # SAFETY CHECK
    assert len(all_preds) == len(all_labels) and len(all_preds) > 0, \
        f"Invalid test data: preds={len(all_preds)}, labels={len(all_labels)}"

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    test_f1_micro = f1_score(all_labels, all_preds, average="micro")
    
    try:
        if args.data in ["mosi", "mimic"]:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        elif args.data == "enrico":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")
        else:
            test_auc = roc_auc_score(all_labels, all_probs)
    except:
        test_auc = 0.0

    total_param = parameter_count(model)[""]
    total_flop = 0

    return (
        best_val_acc, best_val_f1, best_val_auc,
        test_acc, test_f1, test_f1_micro, test_auc,
        train_time / args.train_epochs,
        infer_time,
        total_flop,
        total_param,
    )


# ============================================================================
# REGRESSION TRAINING FUNCTION
# ============================================================================

def train_and_evaluate_imoe_regression(args, seed, fusion_model, fusion):
    """Training function for regression tasks with optional enhanced PID."""
    
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.data == "mosi":
        (
            data_dict, encoder_dict, labels,
            train_ids, val_ids, test_ids,
            n_labels, input_dims, transforms,
            masks, observed_idx_arr, _, _
        ) = load_and_preprocess_data_mosi_regression(args)
    else:
        raise NotImplementedError(f"Regression not supported for {args.data}")

    train_loader, val_loader, test_loader = create_loaders(
        data_dict, observed_idx_arr, labels,
        train_ids, val_ids, test_ids,
        args.batch_size, args.num_workers,
        args.pin_memory, input_dims,
        transforms, masks, args.use_common_ids,
        dataset=args.data
    )

    model = InteractionMoERegression(
        num_modalities=len(args.modality),
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=False,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    # Decomposition (optional)
    decomp = None
    if getattr(args, 'use_info_decomposition', False):
        dims = [input_dims[m] for m in sorted(input_dims)]
        decomp = InfoDecompositionPreprocessor(dims, args.hidden_dim).to(device)

    params = list(model.parameters())
    for enc in encoder_dict.values():
        params += list(enc.parameters())
    if decomp:
        params += list(decomp.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()

    best_val_mae = float('inf')
    best_state = None
    train_time = 0.0

    for epoch in trange(args.train_epochs):
        start = time.time()
        model.train()
        for enc in encoder_dict.values():
            enc.train()
        if decomp:
            decomp.train()

        for samples, batch_labels, *_ in train_loader:
            batch_labels = batch_labels.to(device).float()
            samples = {k: v.to(device) for k, v in samples.items()}
            optimizer.zero_grad()

            fusion_input = [
                ensure_2d(encoder_dict[m](samples[m]))
                for m in samples
            ]

            outputs, _ = moe_forward_per_sample(model, fusion_input)
            loss = criterion(outputs.squeeze(), batch_labels)

            if decomp:
                raw = [ensure_2d(samples[m]) for m in sorted(samples)]
                _, (u, r, s) = decomp(raw)
                loss = loss + getattr(args, 'decomposition_loss_weight', 0.1) * decomposition_loss(u, r, s)

            loss.backward()
            optimizer.step()

        train_time += time.time() - start

        # Validation
        model.eval()
        preds, gts = [], []
        
        with torch.no_grad():
            for samples, batch_labels, *_ in val_loader:
                batch_labels = batch_labels.to(device)
                samples = {k: v.to(device) for k, v in samples.items()}

                fusion_input = [
                    ensure_2d(encoder_dict[m](samples[m]))
                    for m in samples
                ]

                outputs, _ = moe_forward_per_sample(model, fusion_input)
                preds.extend(outputs.squeeze().cpu().numpy())
                gts.extend(batch_labels.cpu().numpy())

        val_mae = mean_absolute_error(gts, preds)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = deepcopy(model.state_dict())

    # Test
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_samples, batch_ids, batch_labels, batch_mcs, batch_observed = batch
            batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device)

            fusion_input = [
                ensure_2d(encoder_dict[m](batch_samples[m]))
                for m in batch_samples
            ]

            outputs, _ = moe_forward_per_sample(model, fusion_input)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    test_mae = mean_absolute_error(all_labels, all_preds)
    test_corr = np.corrcoef(all_labels, all_preds)[0, 1] if len(all_labels) > 1 else 0

    total_param = parameter_count(model)[""]

    return (
        best_val_mae, test_mae, test_corr,
        train_time / args.train_epochs,
        0, 0, total_param,
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_pid_arguments(parser):
    """Add PID arguments to argument parser."""
    
    # Simple PID
    parser.add_argument("--use_info_decomposition", action="store_true",
                        help="Use simple PID module")
    parser.add_argument("--decomposition_loss_weight", type=float, default=1.0,
                        help="Weight for decomposition loss")
    
    # Enhanced PID
    parser.add_argument("--use_enhanced_pid", action="store_true",
                        help="Use enhanced PID module")
    parser.add_argument("--use_bottleneck", action="store_true", default=True,
                        help="Use variational bottleneck in enhanced PID")
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use cross-modal attention in enhanced PID")
    parser.add_argument("--use_aux_predictor", action="store_true", default=False,
                        help="Use auxiliary predictor in enhanced PID")
    parser.add_argument("--pid_kl_weight", type=float, default=0.001,
                        help="Weight for KL loss in enhanced PID")
    parser.add_argument("--pid_aux_weight", type=float, default=0.3,
                        help="Weight for auxiliary loss in enhanced PID")
    
    return parser