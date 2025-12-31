# # src/imoe/imoe_train.py
# # ============================================================================
# # Unified iMoE Training (SAFE, STANDALONE, DROP-IN REPLACEMENT)
# # ============================================================================
# # - Does NOT modify InteractionMoE / InteractionMoERegression
# # - Fixes batch collapse via per-sample adapter
# # - Fully compatible with existing scripts, logging, plots, saves
# # ============================================================================

# import torch
# import torch.nn as nn
# import numpy as np
# import time
# from copy import deepcopy
# from pathlib import Path
# from datetime import datetime
# from tqdm import trange
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
# from fvcore.nn import parameter_count

# # ===================== DATA =====================
# from src.common.datasets.adni import load_and_preprocess_data_adni
# from src.common.datasets.mimic import load_and_preprocess_data_mimic
# from src.common.datasets.enrico import load_and_preprocess_data_enrico
# from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
# from src.common.datasets.mosi import (
#     load_and_preprocess_data_mosi,
#     load_and_preprocess_data_mosi_regression,
# )
# from src.common.datasets.MultiModalDataset import create_loaders

# # ===================== UTILS =====================
# from src.common.utils import (
#     seed_everything,
#     plot_total_loss_curves,
#     plot_interaction_loss_curves,
#     visualize_sample_weights,
#     visualize_expert_logits,
#     visualize_expert_logits_distribution,
#     set_style,
# )

# # ===================== MODELS =====================
# from src.imoe.InteractionMoE import InteractionMoE
# from src.imoe.InteractionMoERegression import InteractionMoERegression

# set_style()

# # ============================================================================
# # SHAPE SAFETY
# # ============================================================================

# def ensure_2d(x: torch.Tensor) -> torch.Tensor:
#     if x.dim() == 1:
#         return x.unsqueeze(0)
#     if x.dim() == 2:
#         return x
#     if x.dim() > 2:
#         return x.mean(dim=tuple(range(1, x.dim() - 1)))
#     raise RuntimeError(f"Invalid tensor shape: {x.shape}")

# # ============================================================================
# # SAFE MoE ADAPTER (THE CORE FIX)
# # ============================================================================

# def moe_forward_per_sample(model, fusion_input):
#     """
#     Calls InteractionMoE one sample at a time.
#     Guarantees correct [B, C] outputs and valid interaction losses.
#     """
#     B = fusion_input[0].shape[0]

#     outputs = []
#     interaction_losses_sum = None

#     for b in range(B):
#         single = [x[b:b+1] for x in fusion_input]
#         _, _, y, inter_losses = model(single)

#         outputs.append(y)

#         if interaction_losses_sum is None:
#             interaction_losses_sum = inter_losses
#         else:
#             interaction_losses_sum = [
#                 a + b for a, b in zip(interaction_losses_sum, inter_losses)
#             ]

#     outputs = torch.cat(outputs, dim=0)
#     interaction_losses = [l / B for l in interaction_losses_sum]

#     return outputs, interaction_losses

# # ============================================================================
# # INFORMATION DECOMPOSITION (OPTIONAL)
# # ============================================================================

# class ModalityDecompositionBranch(nn.Module):
#     def __init__(self, in_dim, hidden_dim, num_other):
#         super().__init__()
#         self.unique = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.synergy = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.redundant = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
#             ) for _ in range(num_other)
#         ])

#     def forward(self, x):
#         u = self.unique(x)
#         s = self.synergy(x)
#         r = [m(x) for m in self.redundant]
#         r = torch.stack(r).mean(dim=0) if r else torch.zeros_like(u)
#         return u, r, s

# class InfoDecompositionPreprocessor(nn.Module):
#     def __init__(self, modality_dims, hidden_dim):
#         super().__init__()
#         self.branches = nn.ModuleList([
#             ModalityDecompositionBranch(d, hidden_dim, len(modality_dims) - 1)
#             for d in modality_dims
#         ])
#         self.restore = nn.ModuleList([
#             nn.Linear(hidden_dim, d) for d in modality_dims
#         ])

#     def forward(self, features):
#         processed = []
#         u_all, r_all, s_all = [], [], []

#         for x, b, r in zip(features, self.branches, self.restore):
#             u, red, syn = b(x)
#             u_all.append(u)
#             r_all.append(red)
#             s_all.append(syn)
#             processed.append(r(u + red + syn))

#         return processed, (
#             torch.stack(u_all).mean(0),
#             torch.stack(r_all).mean(0),
#             torch.stack(s_all).mean(0),
#         )

# def decomposition_loss(u, r, s):
#     return torch.norm(u.T @ r, p="fro") + torch.norm(u.T @ s, p="fro")

# # ============================================================================
# # MAIN TRAINING FUNCTION (DROP-IN)
# # ============================================================================


# def train_and_evaluate_imoe(args, seed, fusion_model, fusion):
#     """
#     SAFE, DROP-IN replacement for original train_and_evaluate_imoe.
#     Preserves full API contract and fixes InteractionMoE batch collapse.
#     """

#     seed_everything(seed)
#     device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
#     print(device)

#     # ======================================================
#     # DATA LOADING (unchanged)
#     # ======================================================
#     if args.data == "mosi":
#         (
#             data_dict, encoder_dict, labels,
#             train_ids, val_ids, test_ids,
#             n_labels, input_dims, transforms,
#             masks, observed_idx_arr, _, _
#         ) = load_and_preprocess_data_mosi(args)
#     else:
#         raise NotImplementedError("Dataset not included in this rewrite")

#     train_loader, val_loader, test_loader = create_loaders(
#         data_dict, observed_idx_arr, labels,
#         train_ids, val_ids, test_ids,
#         args.batch_size, args.num_workers,
#         args.pin_memory, input_dims,
#         transforms, masks, args.use_common_ids,
#         dataset=args.data
#     )

#     # ======================================================
#     # MODEL
#     # ======================================================
#     model = InteractionMoE(
#         num_modalities=len(args.modality),
#         fusion_model=deepcopy(fusion_model),
#         fusion_sparse=False,
#         hidden_dim=args.hidden_dim,
#         hidden_dim_rw=args.hidden_dim_rw,
#         num_layer_rw=args.num_layer_rw,
#         temperature_rw=args.temperature_rw,
#     ).to(device)

#     # ======================================================
#     # OPTIONAL DECOMPOSITION
#     # ======================================================
#     decomp = None
#     if args.use_info_decomposition:
#         dims = [input_dims[m] for m in sorted(input_dims)]
#         decomp = InfoDecompositionPreprocessor(dims, args.hidden_dim).to(device)

#     # ======================================================
#     # OPTIMIZER
#     # ======================================================
#     params = list(model.parameters())
#     for enc in encoder_dict.values():
#         params += list(enc.parameters())
#     if decomp:
#         params += list(decomp.parameters())

#     optimizer = torch.optim.Adam(params, lr=args.lr)
#     criterion = nn.CrossEntropyLoss()

#     # ======================================================
#     # TRACKING
#     # ======================================================
#     best_val_acc = 0.0
#     best_val_f1 = 0.0
#     best_val_auc = 0.0
#     best_state = None

#     train_time = 0.0
#     infer_time = 0.0

#     # ======================================================
#     # TRAINING LOOP
#     # ======================================================
#     for epoch in trange(args.train_epochs):
#         start = time.time()
#         model.train()
#         for enc in encoder_dict.values():
#             enc.train()
#         if decomp:
#             decomp.train()

#         for samples, labels, *_ in train_loader:
#             labels = labels.to(device)
#             samples = {k: v.to(device) for k, v in samples.items()}
#             optimizer.zero_grad()

#             fusion_input = [
#                 ensure_2d(encoder_dict[m](samples[m]))
#                 for m in samples
#             ]

#             outputs, inter_losses = moe_forward_per_sample(model, fusion_input)
#             loss = criterion(outputs, labels)

#             if decomp:
#                 raw = [ensure_2d(samples[m]) for m in sorted(samples)]
#                 _, (u, r, s) = decomp(raw)
#                 loss = loss + args.decomposition_loss_weight * decomposition_loss(u, r, s)

#             loss.backward()
#             optimizer.step()

#         train_time += time.time() - start

#         # ==================================================
#         # VALIDATION
#         # ==================================================
#         model.eval()
#         preds, gts, probs = [], [], []

#         with torch.no_grad():
#             for samples, labels, *_ in val_loader:
#                 labels = labels.to(device)
#                 samples = {k: v.to(device) for k, v in samples.items()}

#                 fusion_input = [
#                     ensure_2d(encoder_dict[m](samples[m]))
#                     for m in samples
#                 ]

#                 outputs, _ = moe_forward_per_sample(model, fusion_input)

#                 p = torch.softmax(outputs, dim=1)
#                 preds.extend(outputs.argmax(1).cpu().numpy())
#                 probs.extend(p[:, 1].cpu().numpy())
#                 gts.extend(labels.cpu().numpy())

#         val_acc = accuracy_score(gts, preds)
#         val_f1 = f1_score(gts, preds, average="macro")
#         val_auc = roc_auc_score(gts, probs)

#         print(
#             f"[Epoch {epoch+1}/{args.train_epochs}] "
#             f"Val Acc: {val_acc*100:.2f} | "
#             f"Val F1: {val_f1*100:.2f} | "
#             f"Val AUC: {val_auc*100:.2f}"
#         )

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_val_f1 = val_f1
#             best_val_auc = val_auc
#             best_state = deepcopy(model.state_dict())

#     # ======================================================
#     # LOAD BEST MODEL
#     # ======================================================
#     model.load_state_dict(best_state)
#     model.eval()

#     # ======================
#     # TEST EVALUATION (FIXED)
#     # ======================
#     model.eval()

#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in test_loader:
#             # MOSI test loader structure
#             batch_samples, batch_ids, batch_labels, batch_mcs, batch_observed = batch

#             batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
#             batch_labels = batch_labels.to(device)

#             fusion_input = [
#                 ensure_2d(encoder_dict[m](batch_samples[m]))
#                 for m in batch_samples
#             ]

#             outputs, _ = moe_forward_per_sample(model, fusion_input)

#             preds = outputs.argmax(dim=1)

#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch_labels.cpu().numpy())

#     # SAFETY CHECK (important)
#     assert len(all_preds) == len(all_labels) and len(all_preds) > 0, \
#         f"Invalid test data: preds={len(all_preds)}, labels={len(all_labels)}"

#     test_acc = accuracy_score(all_labels, all_preds)
#     test_f1 = f1_score(all_labels, all_preds, average="macro")
#     test_f1_micro = f1_score(all_labels, all_preds, average="micro")
#     if args.data in ["mosi", "mimic"]:
#     # binary AUC (use positive-class probability)
#         # test_auc = 0.0 # temporary 
#         probs = np.asarray(probs)
#         if probs.ndim > 1:
#             probs = probs[:, 1]
#         test_auc = roc_auc_score(gts, probs)
#     elif args.data == "adni":
#         test_auc = roc_auc_score(gts, probs, multi_class="ovr")
#     elif args.data == "enrico":
#         test_auc = roc_auc_score(gts, probs, multi_class="ovo")
#     else:
#         test_auc = 0.0

#     total_param = parameter_count(model)[""]
#     total_flop = 0  # unchanged from original unless explicitly computed

#     return (
#         best_val_acc, best_val_f1, best_val_auc,
#         test_acc, test_f1, test_f1_micro, test_auc,
#         train_time / args.train_epochs,
#         infer_time,
#         total_flop,
#         total_param,
#     )

# src/imoe/imoe_train.py
# ============================================================================
# Unified iMoE Training with Enhanced PID Module
# ============================================================================
# Features:
# - SAFE, STANDALONE, DROP-IN REPLACEMENT
# - Does NOT modify InteractionMoE / InteractionMoERegression
# - Fixes batch collapse via per-sample adapter
# - Enhanced Partial Information Decomposition (PID):
#   * Pretrained/Frozen Backbone Support
#   * Mutual Information Neural Estimation (MINE)
#   * Contrastive Learning for Component Disentanglement
#   * Auxiliary Prediction Tasks
#   * Cross-Modal Attention for Synergy/Redundancy
#   * Variational Bottleneck for Information Control
#   * Gradient Reversal for Orthogonality
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

set_style()


# ============================================================================
# SHAPE SAFETY
# ============================================================================

def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D [B, D]."""
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
# GRADIENT REVERSAL LAYER (for adversarial orthogonality)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ============================================================================
# MUTUAL INFORMATION NEURAL ESTIMATION (MINE)
# ============================================================================

class MutualInformationEstimator(nn.Module):
    """
    MINE-based mutual information estimator.
    Uses Donsker-Varadhan representation of KL divergence.
    """
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        self.ma_et = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate MI(X; Y) using MINE."""
        batch_size = x.size(0)
        
        joint = torch.cat([x, y], dim=-1)
        y_shuffle = y[torch.randperm(batch_size)]
        marginal = torch.cat([x, y_shuffle], dim=-1)
        
        t_joint = self.net(joint)
        t_marginal = self.net(marginal)
        
        et = torch.exp(t_marginal - 1)
        if self.ma_et is None:
            self.ma_et = et.mean().detach()
        else:
            self.ma_et = 0.99 * self.ma_et + 0.01 * et.mean().detach()
        
        mi = t_joint.mean() - torch.log(et.mean() + 1e-8)
        return mi


# ============================================================================
# CROSS-MODAL ATTENTION MODULE
# ============================================================================

class CrossModalAttention(nn.Module):
    """Multi-head cross-modal attention for capturing inter-modal relationships."""
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = query.size(0)
        
        q = self.q_proj(query).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(B, self.num_heads, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).reshape(B, -1)
        out = self.out_proj(out)
        
        return self.norm(query + out)


# ============================================================================
# VARIATIONAL INFORMATION BOTTLENECK
# ============================================================================

class VariationalBottleneck(nn.Module):
    """Variational Information Bottleneck for controlled information flow."""
    
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return z, kl.mean()


# ============================================================================
# COMPONENT ENCODERS
# ============================================================================

class UniqueEncoder(nn.Module):
    """Encoder for modality-unique information with adversarial training."""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_other: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.grl = GradientReversalLayer(alpha=0.5)
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, in_dim)
            ) for _ in range(num_other)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def adversarial_loss(
        self, 
        unique_repr: torch.Tensor, 
        other_modalities: List[torch.Tensor]
    ) -> torch.Tensor:
        """Penalize if unique representation can predict other modalities."""
        if not other_modalities:
            return torch.tensor(0.0, device=unique_repr.device)
        
        reversed_repr = self.grl(unique_repr)
        loss = 0.0
        for disc, other in zip(self.discriminators, other_modalities):
            pred = disc(reversed_repr)
            loss += F.mse_loss(pred, other)
        return loss / len(other_modalities)


class RedundancyEncoder(nn.Module):
    """Encoder for cross-modal redundant information using attention."""
    
    def __init__(self, in_dims: List[int], hidden_dim: int, out_dim: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in in_dims
        ])
        
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads=4)
        
        self.merger = nn.Sequential(
            nn.Linear(hidden_dim * len(in_dims), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.bottleneck = VariationalBottleneck(out_dim, out_dim)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        
        attended = []
        for i, p in enumerate(projected):
            others = torch.stack([o for j, o in enumerate(projected) if j != i]).mean(0)
            attended.append(self.cross_attention(p, others))
        
        merged = torch.cat(attended, dim=-1)
        out = self.merger(merged)
        z, kl = self.bottleneck(out)
        
        return z, kl


class SynergyEncoder(nn.Module):
    """Encoder for synergistic information using multiplicative interactions."""
    
    def __init__(self, in_dims: List[int], hidden_dim: int, out_dim: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(d, hidden_dim) for d in in_dims
        ])
        
        n = len(in_dims)
        self.bilinear = nn.ModuleList([
            nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n * (n - 1) // 2)
        ])
        
        total_pairs = n * (n - 1) // 2
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * n, total_pairs),
            nn.Sigmoid()
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * max(1, total_pairs), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
        self.bottleneck = VariationalBottleneck(out_dim, out_dim)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        
        interactions = []
        idx = 0
        for i in range(len(projected)):
            for j in range(i + 1, len(projected)):
                interactions.append(self.bilinear[idx](projected[i], projected[j]))
                idx += 1
        
        if interactions:
            gate_input = torch.cat(projected, dim=-1)
            gates = self.gate(gate_input)
            gated = torch.stack(interactions, dim=1) * gates.unsqueeze(-1)
            gated = gated.view(gated.size(0), -1)
        else:
            gated = projected[0]
        
        out = self.output(gated)
        z, kl = self.bottleneck(out)
        
        return z, kl


# ============================================================================
# AUXILIARY PREDICTOR
# ============================================================================

class AuxiliaryPredictor(nn.Module):
    """Predicts task labels from each PID component."""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


# ============================================================================
# PRETRAINED BACKBONE WRAPPER
# ============================================================================

class PretrainedBackbone(nn.Module):
    """
    Wrapper for pretrained modality encoders.
    Supports frozen, fine-tune, and adapter modes.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        out_dim: int,
        target_dim: int,
        mode: str = "adapter",
        adapter_dim: int = 64
    ):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        
        if mode == "frozen":
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.proj = nn.Linear(out_dim, target_dim)
            
        elif mode == "adapter":
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.adapter = nn.Sequential(
                nn.Linear(out_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, out_dim)
            )
            self.proj = nn.Linear(out_dim, target_dim)
            
        else:  # finetune
            self.proj = nn.Linear(out_dim, target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.mode == "finetune"):
            h = self.backbone(x)
        
        if self.mode == "adapter":
            h = h + self.adapter(h)
        
        return self.proj(h)


# ============================================================================
# CONTRASTIVE LOSS FOR DISENTANGLEMENT
# ============================================================================

class ContrastivePIDLoss(nn.Module):
    """
    Contrastive loss for proper disentanglement:
    - Unique representations should be dissimilar across modalities
    - Redundant representations should be similar across modalities
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def unique_contrastive(self, unique_reprs: List[torch.Tensor]) -> torch.Tensor:
        """Unique representations should be orthogonal."""
        if len(unique_reprs) < 2:
            return torch.tensor(0.0, device=unique_reprs[0].device)
        
        loss = 0.0
        count = 0
        for i in range(len(unique_reprs)):
            for j in range(i + 1, len(unique_reprs)):
                u_i = F.normalize(unique_reprs[i], dim=-1)
                u_j = F.normalize(unique_reprs[j], dim=-1)
                sim = (u_i * u_j).sum(dim=-1)
                loss += sim.pow(2).mean()
                count += 1
        return loss / max(1, count)

    def redundancy_contrastive(self, redundancy_reprs: List[torch.Tensor]) -> torch.Tensor:
        """Redundant representations should be aligned."""
        if len(redundancy_reprs) < 2:
            return torch.tensor(0.0, device=redundancy_reprs[0].device)
        
        loss = 0.0
        count = 0
        for i in range(len(redundancy_reprs)):
            for j in range(i + 1, len(redundancy_reprs)):
                r_i = F.normalize(redundancy_reprs[i], dim=-1)
                r_j = F.normalize(redundancy_reprs[j], dim=-1)
                sim = (r_i * r_j).sum(dim=-1)
                loss -= sim.mean()
                count += 1
        return loss / max(1, count)

    def forward(
        self,
        unique_reprs: List[torch.Tensor],
        redundancy_reprs: List[torch.Tensor]
    ) -> torch.Tensor:
        return self.unique_contrastive(unique_reprs) + self.redundancy_contrastive(redundancy_reprs)


# ============================================================================
# ENHANCED PID MODULE (MAIN CLASS)
# ============================================================================

class EnhancedPIDModule(nn.Module):
    """
    Enhanced Partial Information Decomposition Module.
    
    Decomposes multimodal features into:
    - U (Unique): Information specific to each modality
    - R (Redundant): Shared information across modalities  
    - S (Synergistic): Information emerging from combination
    """
    
    def __init__(
        self,
        modality_dims: List[int],
        hidden_dim: int,
        output_dim: int,
        num_classes: int,
        pretrained_backbones: Optional[Dict[int, nn.Module]] = None,
        backbone_mode: str = "adapter",
        use_mi_estimation: bool = True,
        use_aux_predictor: bool = True,
        use_contrastive: bool = True,
        mi_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_modalities = len(modality_dims)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.modality_dims = modality_dims
        
        # Pretrained backbones (optional)
        self.backbones = nn.ModuleDict()
        self.effective_dims = []
        
        for i, dim in enumerate(modality_dims):
            if pretrained_backbones and i in pretrained_backbones:
                self.backbones[str(i)] = PretrainedBackbone(
                    pretrained_backbones[i],
                    out_dim=dim,
                    target_dim=hidden_dim,
                    mode=backbone_mode
                )
                self.effective_dims.append(hidden_dim)
            else:
                self.backbones[str(i)] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU()
                )
                self.effective_dims.append(hidden_dim)
        
        # Component encoders
        self.unique_encoders = nn.ModuleList([
            UniqueEncoder(hidden_dim, hidden_dim, output_dim, self.num_modalities - 1)
            for _ in range(self.num_modalities)
        ])
        
        self.redundancy_encoder = RedundancyEncoder(
            [hidden_dim] * self.num_modalities, hidden_dim, output_dim
        )
        
        self.synergy_encoder = SynergyEncoder(
            [hidden_dim] * self.num_modalities, hidden_dim, output_dim
        )
        
        # Restoration layers
        self.restore = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for dim in modality_dims
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * (self.num_modalities + 2), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Optional: Mutual Information estimators
        self.use_mi = use_mi_estimation
        if use_mi_estimation:
            self.mi_unique_red = nn.ModuleList([
                MutualInformationEstimator(output_dim, output_dim, mi_hidden_dim)
                for _ in range(self.num_modalities)
            ])
            self.mi_unique_syn = nn.ModuleList([
                MutualInformationEstimator(output_dim, output_dim, mi_hidden_dim)
                for _ in range(self.num_modalities)
            ])
        
        # Optional: Auxiliary predictors
        self.use_aux = use_aux_predictor
        if use_aux_predictor:
            self.aux_unique = nn.ModuleList([
                AuxiliaryPredictor(output_dim, hidden_dim // 2, num_classes)
                for _ in range(self.num_modalities)
            ])
            self.aux_redundant = AuxiliaryPredictor(output_dim, hidden_dim // 2, num_classes)
            self.aux_synergy = AuxiliaryPredictor(output_dim, hidden_dim // 2, num_classes)
        
        # Optional: Contrastive loss
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_loss = ContrastivePIDLoss()
        
        # Learnable temperature for component weighting
        self.component_temp = nn.Parameter(torch.ones(3))
        
        # Store last components for analysis
        self._last_components = None

    def forward(
        self,
        features: List[torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features: List of modality features [B, D_i] for each modality
            labels: Optional labels for auxiliary loss computation
            
        Returns:
            processed: Restored features for each modality
            fused: Fused representation [B, output_dim]
            losses: Dictionary of component losses
        """
        losses = {}
        device = features[0].device
        
        # Apply backbones
        encoded = [self.backbones[str(i)](f) for i, f in enumerate(features)]
        
        # Unique information
        unique_reprs = []
        for i, enc in enumerate(self.unique_encoders):
            u = enc(encoded[i])
            unique_reprs.append(u)
            
            others = [encoded[j] for j in range(len(encoded)) if j != i]
            if others:
                adv_loss = enc.adversarial_loss(u, others)
                losses[f"adv_unique_{i}"] = adv_loss
        
        # Redundant information
        redundant, kl_red = self.redundancy_encoder(encoded)
        losses["kl_redundant"] = kl_red
        
        # Synergistic information
        synergy, kl_syn = self.synergy_encoder(encoded)
        losses["kl_synergy"] = kl_syn
        
        # MI minimization losses
        if self.use_mi:
            for i, (mi_ur, mi_us) in enumerate(zip(self.mi_unique_red, self.mi_unique_syn)):
                losses[f"mi_u{i}_r"] = mi_ur(unique_reprs[i], redundant)
                losses[f"mi_u{i}_s"] = mi_us(unique_reprs[i], synergy)
        
        # Contrastive losses
        if self.use_contrastive:
            losses["contrastive"] = self.contrastive_loss(
                unique_reprs, 
                [redundant] * len(unique_reprs)
            )
        
        # Auxiliary prediction losses
        if self.use_aux and labels is not None:
            aux_criterion = nn.CrossEntropyLoss()
            for i, aux in enumerate(self.aux_unique):
                losses[f"aux_unique_{i}"] = aux_criterion(aux(unique_reprs[i]), labels)
            losses["aux_redundant"] = aux_criterion(self.aux_redundant(redundant), labels)
            losses["aux_synergy"] = aux_criterion(self.aux_synergy(synergy), labels)
        
        # Restore original dimensions
        processed = []
        for i, restore in enumerate(self.restore):
            combined = torch.cat([unique_reprs[i], redundant, synergy], dim=-1)
            processed.append(restore(combined))
        
        # Fused representation with learned temperature
        weights = F.softmax(self.component_temp, dim=0)
        all_components = [u * weights[0] for u in unique_reprs]
        all_components.append(redundant * weights[1])
        all_components.append(synergy * weights[2])
        
        fused = self.fusion(torch.cat(all_components, dim=-1))
        
        # Store for analysis
        self._last_components = {
            "unique": torch.stack(unique_reprs),
            "redundant": redundant,
            "synergy": synergy,
            "weights": weights.detach()
        }
        
        return processed, fused, losses

    def get_component_importance(self) -> Dict[str, float]:
        """Get learned importance weights for each component."""
        weights = F.softmax(self.component_temp, dim=0)
        return {
            "unique": weights[0].item(),
            "redundant": weights[1].item(),
            "synergy": weights[2].item()
        }

    def get_last_components(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the last computed components for analysis."""
        return self._last_components


# ============================================================================
# SIMPLE PID MODULE (LEGACY COMPATIBILITY)
# ============================================================================

class ModalityDecompositionBranch(nn.Module):
    """Simple decomposition branch (legacy)."""
    
    def __init__(self, in_dim: int, hidden_dim: int, num_other: int):
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.unique(x)
        s = self.synergy(x)
        r = [m(x) for m in self.redundant]
        r = torch.stack(r).mean(dim=0) if r else torch.zeros_like(u)
        return u, r, s


class InfoDecompositionPreprocessor(nn.Module):
    """Simple info decomposition preprocessor (legacy)."""
    
    def __init__(self, modality_dims: List[int], hidden_dim: int):
        super().__init__()
        self.branches = nn.ModuleList([
            ModalityDecompositionBranch(d, hidden_dim, len(modality_dims) - 1)
            for d in modality_dims
        ])
        self.restore = nn.ModuleList([
            nn.Linear(hidden_dim, d) for d in modality_dims
        ])

    def forward(
        self, 
        features: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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


def decomposition_loss(u: torch.Tensor, r: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Simple orthogonality loss (legacy)."""
    return torch.norm(u.T @ r, p="fro") + torch.norm(u.T @ s, p="fro")


# ============================================================================
# ENHANCED DECOMPOSITION LOSS
# ============================================================================

def compute_enhanced_decomposition_loss(
    pid_losses: Dict[str, torch.Tensor],
    config: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Compute weighted sum of all PID-related losses.
    """
    default_config = {
        "adv": 0.1,
        "kl": 0.01,
        "mi": 0.1,
        "contrastive": 0.1,
        "aux": 0.5,
    }
    
    config = {**default_config, **(config or {})}
    
    if not pid_losses:
        return torch.tensor(0.0)
    
    device = next(iter(pid_losses.values())).device
    total_loss = torch.tensor(0.0, device=device)
    
    for key, loss in pid_losses.items():
        if key.startswith("adv"):
            total_loss = total_loss + config["adv"] * loss
        elif key.startswith("kl"):
            total_loss = total_loss + config["kl"] * loss
        elif key.startswith("mi"):
            total_loss = total_loss + config["mi"] * loss
        elif key == "contrastive":
            total_loss = total_loss + config["contrastive"] * loss
        elif key.startswith("aux"):
            total_loss = total_loss + config["aux"] * loss
    
    return total_loss


# ============================================================================
# MAIN TRAINING FUNCTION (DROP-IN REPLACEMENT)
# ============================================================================

def train_and_evaluate_imoe(
    args, 
    seed: int, 
    fusion_model: nn.Module, 
    fusion: Any
) -> Tuple:
    """
    SAFE, DROP-IN replacement for original train_and_evaluate_imoe.
    Preserves full API contract and fixes InteractionMoE batch collapse.
    
    Supports both legacy and enhanced PID modes.
    """
    
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======================================================
    # DATA LOADING
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

    # Move encoders to device
    for key in encoder_dict:
        encoder_dict[key] = encoder_dict[key].to(device)

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
    # PID MODULE SETUP
    # ======================================================
    pid_module = None
    use_enhanced_pid = getattr(args, 'use_enhanced_pid', False)
    use_simple_pid = getattr(args, 'use_info_decomposition', False)
    
    if use_enhanced_pid:
        dims = [input_dims[m] for m in sorted(input_dims)]
        
        # Optional: Load pretrained backbones
        pretrained = None
        if hasattr(args, 'pretrained_backbones') and args.pretrained_backbones:
            pretrained = {}
            for i, path in enumerate(args.pretrained_backbones):
                if path and Path(path).exists():
                    backbone = torch.load(path, map_location=device)
                    pretrained[i] = backbone
        
        pid_module = EnhancedPIDModule(
            modality_dims=dims,
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_classes=n_labels,
            pretrained_backbones=pretrained,
            backbone_mode=getattr(args, 'backbone_mode', 'adapter'),
            use_mi_estimation=getattr(args, 'use_mi_estimation', True),
            use_aux_predictor=getattr(args, 'use_aux_predictor', True),
            use_contrastive=getattr(args, 'use_contrastive', True),
        ).to(device)
        print("Using Enhanced PID Module")
        
    elif use_simple_pid:
        dims = [input_dims[m] for m in sorted(input_dims)]
        pid_module = InfoDecompositionPreprocessor(dims, args.hidden_dim).to(device)
        print("Using Simple PID Module (Legacy)")

    # ======================================================
    # OPTIMIZER
    # ======================================================
    params = list(model.parameters())
    for enc in encoder_dict.values():
        params += list(enc.parameters())
    if pid_module:
        params += list(pid_module.parameters())

    optimizer = torch.optim.AdamW(
        params, 
        lr=args.lr, 
        weight_decay=getattr(args, 'weight_decay', 0.01)
    )
    
    # Optional scheduler
    use_scheduler = getattr(args, 'use_scheduler', True)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.train_epochs, 
            eta_min=args.lr * 0.01
        )
    
    criterion = nn.CrossEntropyLoss()

    # ======================================================
    # TRACKING
    # ======================================================
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_state = None
    best_encoder_states = None
    best_pid_state = None

    train_time = 0.0
    infer_time = 0.0
    
    train_losses = []
    val_metrics = []

    # PID loss weights
    pid_config = {
        "adv": getattr(args, 'pid_adv_weight', 0.1),
        "kl": getattr(args, 'pid_kl_weight', 0.01),
        "mi": getattr(args, 'pid_mi_weight', 0.1),
        "contrastive": getattr(args, 'pid_contrastive_weight', 0.1),
        "aux": getattr(args, 'pid_aux_weight', 0.5),
    }
    decomp_weight = getattr(args, 'decomposition_loss_weight', 0.1)

    # ======================================================
    # TRAINING LOOP
    # ======================================================
    for epoch in trange(args.train_epochs, desc="Training"):
        start = time.time()
        model.train()
        for enc in encoder_dict.values():
            enc.train()
        if pid_module:
            pid_module.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_loader:
            # Handle different batch formats
            if len(batch_data) == 2:
                samples, batch_labels = batch_data
            else:
                samples, batch_labels = batch_data[0], batch_data[2] if len(batch_data) > 2 else batch_data[1]
            
            batch_labels = batch_labels.to(device)
            samples = {k: v.to(device) for k, v in samples.items()}
            optimizer.zero_grad()

            # Encode
            fusion_input = [
                ensure_2d(encoder_dict[m](samples[m]))
                for m in samples
            ]

            # Apply PID module
            pid_losses = {}
            if isinstance(pid_module, EnhancedPIDModule):
                processed, fused_pid, pid_losses = pid_module(fusion_input, batch_labels)
                fusion_input = processed
            elif isinstance(pid_module, InfoDecompositionPreprocessor):
                raw = [ensure_2d(samples[m]) for m in sorted(samples)]
                processed, (u, r, s) = pid_module(raw)
                fusion_input = [
                    ensure_2d(encoder_dict[m](samples[m]))
                    for m in samples
                ]

            # MoE forward
            outputs, inter_losses = moe_forward_per_sample(model, fusion_input)
            
            # Main loss
            loss = criterion(outputs, batch_labels)

            # PID losses
            if isinstance(pid_module, EnhancedPIDModule):
                pid_loss = compute_enhanced_decomposition_loss(pid_losses, pid_config)
                loss = loss + pid_loss
            elif isinstance(pid_module, InfoDecompositionPreprocessor):
                loss = loss + decomp_weight * decomposition_loss(u, r, s)

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        if use_scheduler:
            scheduler.step()

        train_time += time.time() - start
        train_losses.append(epoch_loss / num_batches)

        # Log component importance
        if isinstance(pid_module, EnhancedPIDModule) and epoch % 10 == 0:
            importance = pid_module.get_component_importance()
            print(f"\n[Epoch {epoch+1}] Component weights - "
                  f"U: {importance['unique']:.3f}, "
                  f"R: {importance['redundant']:.3f}, "
                  f"S: {importance['synergy']:.3f}")

        # ==================================================
        # VALIDATION
        # ==================================================
        model.eval()
        for enc in encoder_dict.values():
            enc.eval()
        if pid_module:
            pid_module.eval()
            
        preds, gts, probs = [], [], []

        infer_start = time.time()
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:
                    samples, batch_labels = batch_data
                else:
                    samples, batch_labels = batch_data[0], batch_data[2] if len(batch_data) > 2 else batch_data[1]
                
                batch_labels = batch_labels.to(device)
                samples = {k: v.to(device) for k, v in samples.items()}

                fusion_input = [
                    ensure_2d(encoder_dict[m](samples[m]))
                    for m in samples
                ]

                if isinstance(pid_module, EnhancedPIDModule):
                    processed, _, _ = pid_module(fusion_input, None)
                    fusion_input = processed

                outputs, _ = moe_forward_per_sample(model, fusion_input)

                p = torch.softmax(outputs, dim=1)
                preds.extend(outputs.argmax(1).cpu().numpy())
                if n_labels == 2:
                    probs.extend(p[:, 1].cpu().numpy())
                else:
                    probs.extend(p.cpu().numpy())
                gts.extend(batch_labels.cpu().numpy())

        infer_time += time.time() - infer_start

        val_acc = accuracy_score(gts, preds)
        val_f1 = f1_score(gts, preds, average="macro")
        
        try:
            if n_labels == 2:
                val_auc = roc_auc_score(gts, probs)
            else:
                val_auc = roc_auc_score(gts, probs, multi_class="ovr")
        except:
            val_auc = 0.0

        val_metrics.append({
            'acc': val_acc,
            'f1': val_f1,
            'auc': val_auc
        })

        if epoch % 5 == 0 or epoch == args.train_epochs - 1:
            print(
                f"\n[Epoch {epoch+1}/{args.train_epochs}] "
                f"Loss: {train_losses[-1]:.4f} | "
                f"Val Acc: {val_acc*100:.2f} | "
                f"Val F1: {val_f1*100:.2f} | "
                f"Val AUC: {val_auc*100:.2f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
            best_state = deepcopy(model.state_dict())
            best_encoder_states = {k: deepcopy(v.state_dict()) for k, v in encoder_dict.items()}
            if pid_module:
                best_pid_state = deepcopy(pid_module.state_dict())

    # ======================================================
    # LOAD BEST MODEL
    # ======================================================
    model.load_state_dict(best_state)
    for k, v in encoder_dict.items():
        v.load_state_dict(best_encoder_states[k])
    if pid_module and best_pid_state:
        pid_module.load_state_dict(best_pid_state)

    # ======================================================
    # TEST EVALUATION
    # ======================================================
    model.eval()
    for enc in encoder_dict.values():
        enc.eval()
    if pid_module:
        pid_module.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_data in test_loader:
            # Handle various batch formats
            if len(batch_data) == 5:
                batch_samples, batch_ids, batch_labels, batch_mcs, batch_observed = batch_data
            elif len(batch_data) == 2:
                batch_samples, batch_labels = batch_data
            else:
                batch_samples = batch_data[0]
                batch_labels = batch_data[2] if len(batch_data) > 2 else batch_data[1]

            batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device)

            fusion_input = [
                ensure_2d(encoder_dict[m](batch_samples[m]))
                for m in batch_samples
            ]

            if isinstance(pid_module, EnhancedPIDModule):
                processed, _, _ = pid_module(fusion_input, None)
                fusion_input = processed

            outputs, _ = moe_forward_per_sample(model, fusion_input)

            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            if n_labels == 2:
                all_probs.extend(probs[:, 1].cpu().numpy())
            else:
                all_probs.extend(probs.cpu().numpy())

    # SAFETY CHECK
    assert len(all_preds) == len(all_labels) and len(all_preds) > 0, \
        f"Invalid test data: preds={len(all_preds)}, labels={len(all_labels)}"

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    test_f1_micro = f1_score(all_labels, all_preds, average="micro")
    
    try:
        if n_labels == 2:
            test_auc = roc_auc_score(all_labels, all_probs)
        elif args.data == "adni":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        elif args.data == "enrico":
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovo")
        else:
            test_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
    except:
        test_auc = 0.0

    print(f"\n{'='*50}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 (macro): {test_f1*100:.2f}%")
    print(f"Test F1 (micro): {test_f1_micro*100:.2f}%")
    print(f"Test AUC: {test_auc*100:.2f}%")
    print(f"{'='*50}")

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

def train_and_evaluate_imoe_regression(
    args, 
    seed: int, 
    fusion_model: nn.Module, 
    fusion: Any
) -> Tuple:
    """
    Training function for regression tasks with Enhanced PID support.
    """
    
    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ======================================================
    # DATA LOADING
    # ======================================================
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

    for key in encoder_dict:
        encoder_dict[key] = encoder_dict[key].to(device)

    # ======================================================
    # MODEL
    # ======================================================
    model = InteractionMoERegression(
        num_modalities=len(args.modality),
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=False,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    # ======================================================
    # PID MODULE
    # ======================================================
    pid_module = None
    use_enhanced_pid = getattr(args, 'use_enhanced_pid', False)
    
    if use_enhanced_pid:
        dims = [input_dims[m] for m in sorted(input_dims)]
        pid_module = EnhancedPIDModule(
            modality_dims=dims,
            hidden_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            num_classes=1,  # Regression
            use_aux_predictor=False,  # No aux for regression
            use_mi_estimation=getattr(args, 'use_mi_estimation', True),
            use_contrastive=getattr(args, 'use_contrastive', True),
        ).to(device)

    # ======================================================
    # OPTIMIZER
    # ======================================================
    params = list(model.parameters())
    for enc in encoder_dict.values():
        params += list(enc.parameters())
    if pid_module:
        params += list(pid_module.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.train_epochs, eta_min=args.lr * 0.01
    )
    criterion = nn.MSELoss()

    # ======================================================
    # TRACKING
    # ======================================================
    best_val_mae = float('inf')
    best_state = None
    train_time = 0.0

    pid_config = {
        "adv": getattr(args, 'pid_adv_weight', 0.1),
        "kl": getattr(args, 'pid_kl_weight', 0.01),
        "mi": getattr(args, 'pid_mi_weight', 0.1),
        "contrastive": getattr(args, 'pid_contrastive_weight', 0.1),
        "aux": 0.0,  # No aux for regression
    }

    # ======================================================
    # TRAINING LOOP
    # ======================================================
    for epoch in trange(args.train_epochs, desc="Training"):
        start = time.time()
        model.train()
        for enc in encoder_dict.values():
            enc.train()
        if pid_module:
            pid_module.train()

        for batch_data in train_loader:
            if len(batch_data) == 2:
                samples, batch_labels = batch_data
            else:
                samples, batch_labels = batch_data[0], batch_data[2]
            
            batch_labels = batch_labels.to(device).float()
            samples = {k: v.to(device) for k, v in samples.items()}
            optimizer.zero_grad()

            fusion_input = [
                ensure_2d(encoder_dict[m](samples[m]))
                for m in samples
            ]

            pid_losses = {}
            if isinstance(pid_module, EnhancedPIDModule):
                processed, _, pid_losses = pid_module(fusion_input, None)
                fusion_input = processed

            outputs, inter_losses = moe_forward_per_sample(model, fusion_input)
            
            loss = criterion(outputs.squeeze(), batch_labels)

            if pid_losses:
                pid_loss = compute_enhanced_decomposition_loss(pid_losses, pid_config)
                loss = loss + pid_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        scheduler.step()
        train_time += time.time() - start

        # Validation
        model.eval()
        preds, gts = [], []
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 2:
                    samples, batch_labels = batch_data
                else:
                    samples, batch_labels = batch_data[0], batch_data[2]
                
                batch_labels = batch_labels.to(device)
                samples = {k: v.to(device) for k, v in samples.items()}

                fusion_input = [
                    ensure_2d(encoder_dict[m](samples[m]))
                    for m in samples
                ]

                if isinstance(pid_module, EnhancedPIDModule):
                    processed, _, _ = pid_module(fusion_input, None)
                    fusion_input = processed

                outputs, _ = moe_forward_per_sample(model, fusion_input)
                
                preds.extend(outputs.squeeze().cpu().numpy())
                gts.extend(batch_labels.cpu().numpy())

        val_mae = mean_absolute_error(gts, preds)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = deepcopy(model.state_dict())

    # Load best and test
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            if len(batch_data) >= 3:
                batch_samples, _, batch_labels = batch_data[:3]
            else:
                batch_samples, batch_labels = batch_data[:2]

            batch_samples = {k: v.to(device) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device)

            fusion_input = [
                ensure_2d(encoder_dict[m](batch_samples[m]))
                for m in batch_samples
            ]

            if isinstance(pid_module, EnhancedPIDModule):
                processed, _, _ = pid_module(fusion_input, None)
                fusion_input = processed

            outputs, _ = moe_forward_per_sample(model, fusion_input)
            
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    test_mae = mean_absolute_error(all_labels, all_preds)
    test_corr = np.corrcoef(all_labels, all_preds)[0, 1]

    print(f"\nTest MAE: {test_mae:.4f}")
    print(f"Test Correlation: {test_corr:.4f}")

    total_param = parameter_count(model)[""]

    return (
        best_val_mae, test_mae, test_corr,
        train_time / args.train_epochs,
        0, 0, total_param,
    )


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_enhanced_pid(
    modality_dims: List[int],
    hidden_dim: int,
    output_dim: int,
    num_classes: int,
    pretrained_backbones: Optional[Dict] = None,
    **kwargs
) -> EnhancedPIDModule:
    """Factory function for creating EnhancedPIDModule."""
    return EnhancedPIDModule(
        modality_dims=modality_dims,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_classes=num_classes,
        pretrained_backbones=pretrained_backbones,
        **kwargs
    )


# ============================================================================
# ARGUMENT EXTENSION HELPER
# ============================================================================

def add_pid_arguments(parser):
    """Add Enhanced PID arguments to argument parser."""
    
    # PID module flags
    parser.add_argument("--use_enhanced_pid", action="store_true",
                        help="Use enhanced PID module")
    parser.add_argument("--use_info_decomposition", action="store_true",
                        help="Use simple/legacy PID module")
    
    # Backbone settings
    parser.add_argument("--backbone_mode", type=str, default="adapter",
                        choices=["frozen", "adapter", "finetune"],
                        help="Mode for pretrained backbones")
    parser.add_argument("--pretrained_backbones", nargs="*", default=None,
                        help="Paths to pretrained backbone weights")
    
    # PID component flags
    parser.add_argument("--use_mi_estimation", action="store_true", default=True,
                        help="Use mutual information estimation")
    parser.add_argument("--use_aux_predictor", action="store_true", default=True,
                        help="Use auxiliary predictors")
    parser.add_argument("--use_contrastive", action="store_true", default=True,
                        help="Use contrastive disentanglement")
    
    # Loss weights
    parser.add_argument("--pid_adv_weight", type=float, default=0.1,
                        help="Weight for adversarial uniqueness loss")
    parser.add_argument("--pid_kl_weight", type=float, default=0.01,
                        help="Weight for KL divergence loss")
    parser.add_argument("--pid_mi_weight", type=float, default=0.1,
                        help="Weight for MI minimization loss")
    parser.add_argument("--pid_contrastive_weight", type=float, default=0.1,
                        help="Weight for contrastive loss")
    parser.add_argument("--pid_aux_weight", type=float, default=0.5,
                        help="Weight for auxiliary prediction loss")
    parser.add_argument("--decomposition_loss_weight", type=float, default=0.1,
                        help="Weight for simple decomposition loss")
    
    # Training settings
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    parser.add_argument("--use_scheduler", action="store_true", default=True,
                        help="Use learning rate scheduler")
    
    return parser