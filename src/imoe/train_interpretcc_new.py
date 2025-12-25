import os
import sys
import argparse
import warnings
import torch
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

# ============================================================
# Core imports (UNCHANGED)
# ============================================================
from src.common.fusion_models.interpretcc import InterpretCC
from src.imoe.imoe_train import train_and_evaluate_imoe
from src.common.utils import setup_logger, str2bool

# ============================================================
# Dataset loaders (UNCHANGED)
# ============================================================
from src.common.datasets.adni import load_and_preprocess_data_adni
from src.common.datasets.mimic import load_and_preprocess_data_mimic
from src.common.datasets.enrico import load_and_preprocess_data_enrico
from src.common.datasets.mmimdb import load_and_preprocess_data_mmimdb
from src.common.datasets.mosi import (
    load_and_preprocess_data_mosi,
    load_and_preprocess_data_mosi_regression,
)
from src.common.datasets.MultiModalDataset import create_loaders


# ============================================================
# Utils
# ============================================================

def safe_cov(x, y):
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    return (x.T @ y) / (x.size(0) - 1 + 1e-6)


# ============================================================
# DnR MODULE (REPRESENTATION ONLY)
# ============================================================

class DnRBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_u = torch.nn.Linear(dim, dim)
        self.to_r = torch.nn.Linear(dim, dim)
        self.to_s = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.to_u(x), self.to_r(x), self.to_s(x)


class DnRModule(torch.nn.Module):
    """
    Input : List[z_m] , z_m ∈ R^{B×D}
    Output: List[(U_m, R_m, S_m)]
    """
    def __init__(self, hidden_dim, num_modalities):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [DnRBlock(hidden_dim) for _ in range(num_modalities)]
        )

    def forward(self, zs):
        return [blk(z) for blk, z in zip(self.blocks, zs)]


def compute_dnr_loss(dnr_outs):
    """
    Loss = uncorrelation (U ⟂ R) + cross-modal correlation (R, S)
    """
    # ---------- U ⟂ R ----------
    L_unco = 0.0
    for u, r, _ in dnr_outs:
        L_unco += torch.abs(safe_cov(u, r)).mean()
    L_unco /= len(dnr_outs)

    # ---------- Cross-modal ----------
    L_cor = 0.0
    cnt = 0
    for i in range(len(dnr_outs)):
        for j in range(i + 1, len(dnr_outs)):
            _, r_i, s_i = dnr_outs[i]
            _, r_j, s_j = dnr_outs[j]
            L_cor += -safe_cov(r_i, r_j).mean()
            L_cor += -safe_cov(s_i, s_j).mean()
            cnt += 2
    L_cor /= max(cnt, 1)

    return L_unco + L_cor


def recompose(dnr_outs):
    """U + R + S"""
    return [u + r + s for (u, r, s) in dnr_outs]


# ============================================================
# Build TRAIN loader + encoder_dict (NO LOGIC CHANGE)
# ============================================================

def build_train_loader(args):
    if args.data == "adni":
        data = load_and_preprocess_data_adni(args)
    elif args.data == "mimic":
        data = load_and_preprocess_data_mimic(args)
    elif args.data == "enrico":
        data = load_and_preprocess_data_enrico(args)
    elif args.data == "mmimdb":
        data = load_and_preprocess_data_mmimdb(args)
    elif args.data == "mosi":
        data = load_and_preprocess_data_mosi(args)
    elif args.data == "mosi_regression":
        data = load_and_preprocess_data_mosi_regression(args)
    else:
        raise ValueError(args.data)

    (
        data_dict,
        encoder_dict,
        labels,
        train_ids,
        valid_ids,
        test_ids,
        _,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        _,
        _,
    ) = data

    train_loader, _, _ = create_loaders(
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

    return train_loader, encoder_dict


# ============================================================
# DnR PRETRAIN (REPRESENTATION-LEVEL, CLEAN)
# ============================================================

def pretrain_dnr(args, device):
    print(">> Pretraining DnR (representation-only)")

    train_loader, encoder_dict = build_train_loader(args)

    encoders = [
        encoder_dict[m].to(device).eval()
        for m in sorted(encoder_dict.keys())
    ]

    for enc in encoders:
        for p in enc.parameters():
            p.requires_grad = False

    dnr = DnRModule(args.hidden_dim, len(encoders)).to(device)
    opt = torch.optim.Adam(dnr.parameters(), lr=1e-3)
    
    total_steps = len(train_loader)

    for _ in range(args.dnr_pretrain_epochs):
        for step, batch in enumerate(train_loader, start=1):
            tensor_dict = batch[0]   # dict[str, Tensor]

            zs = []
            for i, key in enumerate(sorted(tensor_dict.keys())):
                x = tensor_dict[key].to(device)
                with torch.no_grad():
                    z = encoders[i](x)
                zs.append(z)

            outs = dnr(zs)
            loss = compute_dnr_loss(outs)

            opt.zero_grad()
            loss.backward()
            opt.step()
            # if step % 50 == 0 or step == total_steps:
            #     print(f"Step {step}/{total_steps}, Loss: {loss.item()}")
    print(">> DnR pretraining completed.")

    dnr.eval()
    for p in dnr.parameters():
        p.requires_grad = False

    return dnr


# ============================================================
# Args (FULL – KHỚP imoe_train)
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("iMoE-interpretcc + DnR")

    # dataset
    parser.add_argument("--data", type=str, default="adni")
    parser.add_argument("--modality", type=str, default="IGCB")
    parser.add_argument("--patch", type=str2bool, default=False)
    parser.add_argument("--num_patches", type=int, default=16)
    parser.add_argument("--initial_filling", type=str, default="mean")
    parser.add_argument("--use_common_ids", type=str2bool, default=True)

    # system
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--save", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)

    # training
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)

    # imoe-specific
    parser.add_argument("--fusion_sparse", type=str2bool, default=False)
    parser.add_argument("--temperature_rw", type=float, default=1.0)
    parser.add_argument("--hidden_dim_rw", type=int, default=256)
    parser.add_argument("--num_layer_rw", type=int, default=1)
    parser.add_argument("--interaction_loss_weight", type=float, default=1e-2)

    # interpretcc
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--hard", type=str2bool, default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.5)

    # DnR
    parser.add_argument("--use_info_decomposition", type=str2bool, default=False)
    parser.add_argument("--dnr_pretrain_epochs", type=int, default=5)

    return parser.parse_known_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args, _ = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print("Modality:", args.modality)

    data_to_nlabels = {
        "adni": 3,
        "mimic": 2,
        "mmimdb": 23,
        "enrico": 20,
        "mosi": 2,
        "mosi_regression": 1,
    }

    fusion_model = InterpretCC(
        num_classes=data_to_nlabels[args.data],
        num_modality=len(args.modality),
        input_dim=args.hidden_dim,
        dropout=args.dropout,
        tau=args.tau,
        hard=args.hard,
        threshold=args.threshold,
    ).to(device)

    # ========== DnR ==========
    if args.use_info_decomposition:
        dnr = pretrain_dnr(args, device)

        old_forward = fusion_model.forward

        def new_forward(inputs):
            outs = dnr(inputs)
            inputs = recompose(outs)
            return old_forward(inputs)

        fusion_model.forward = new_forward

    # ========== ORIGINAL TRAIN ==========
    train_and_evaluate_imoe(
        args, args.seed, fusion_model, "interpretcc"
    )


if __name__ == "__main__":
    main()