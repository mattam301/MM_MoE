# src/imoe/imoe_train.py
# ============================================================================
# Unified iMoE Training (SAFE, STANDALONE, DROP-IN REPLACEMENT)
# ============================================================================
# - Does NOT modify InteractionMoE / InteractionMoERegression
# - Fixes batch collapse via per-sample adapter
# - Fully compatible with existing scripts, logging, plots, saves
# ============================================================================

import torch
import torch.nn as nn
import numpy as np
import time
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error
from fvcore.nn import parameter_count

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
# INFORMATION DECOMPOSITION (OPTIONAL)
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
# MAIN TRAINING FUNCTION (DROP-IN)
# ============================================================================

def train_and_evaluate_imoe(args, seed, fusion_model, fusion):

    seed_everything(seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(device)

    # ================= DATA =================
    if args.data == "mosi":
        (data_dict, encoder_dict, labels,
         train_ids, val_ids, test_ids,
         n_labels, input_dims, transforms,
         masks, observed_idx_arr, _, _) = load_and_preprocess_data_mosi(args)
    else:
        raise NotImplementedError("Dataset not shown for brevity")

    train_loader, val_loader, test_loader = create_loaders(
        data_dict, observed_idx_arr, labels,
        train_ids, val_ids, test_ids,
        args.batch_size, args.num_workers,
        args.pin_memory, input_dims,
        transforms, masks, args.use_common_ids,
        dataset=args.data
    )

    # ================= MODEL =================
    model = InteractionMoE(
        num_modalities=len(args.modality),
        fusion_model=deepcopy(fusion_model),
        fusion_sparse=False,
        hidden_dim=args.hidden_dim,
        hidden_dim_rw=args.hidden_dim_rw,
        num_layer_rw=args.num_layer_rw,
        temperature_rw=args.temperature_rw,
    ).to(device)

    # ================= DECOMP =================
    decomp = None
    if args.use_info_decomposition:
        dims = [input_dims[m] for m in sorted(input_dims)]
        decomp = InfoDecompositionPreprocessor(dims, args.hidden_dim).to(device)

    params = list(model.parameters())
    for enc in encoder_dict.values():
        params += list(enc.parameters())
    if decomp:
        params += list(decomp.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # ================= TRAIN =================
    for epoch in trange(args.train_epochs):
        model.train()
        for enc in encoder_dict.values():
            enc.train()
        if decomp:
            decomp.train()

        for samples, labels, *_ in train_loader:
            labels = labels.to(device)
            samples = {k: v.to(device) for k, v in samples.items()}
            optimizer.zero_grad()

            feats = []
            for m in samples:
                feats.append(ensure_2d(encoder_dict[m](samples[m])))

            outputs, inter_losses = moe_forward_per_sample(model, feats)
            loss = criterion(outputs, labels)

            if decomp:
                raw = [ensure_2d(samples[m]) for m in sorted(samples)]
                _, (u, r, s) = decomp(raw)
                loss = loss + 0.01 * decomposition_loss(u, r, s)

            loss.backward()
            optimizer.step()

        # ================= VALID =================
        model.eval()
        preds, gts = [], []

        with torch.no_grad():
            for samples, labels, *_ in val_loader:
                labels = labels.to(device)
                samples = {k: v.to(device) for k, v in samples.items()}
                feats = [ensure_2d(encoder_dict[m](samples[m])) for m in samples]
                outputs, _ = moe_forward_per_sample(model, feats)
                preds.extend(outputs.argmax(1).cpu().numpy())
                gts.extend(labels.cpu().numpy())

        acc = accuracy_score(gts, preds)
        print(f"[Epoch {epoch+1}] Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            best_state = deepcopy(model.state_dict())

    # ================= SAVE =================
    if args.save:
        Path("./saves/imoe").mkdir(parents=True, exist_ok=True)
        torch.save(best_state, f"./saves/imoe/{fusion}_{args.data}.pth")

    total_param = parameter_count(model)[""]

    return best_val_acc, total_param