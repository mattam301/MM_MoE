"""
Information-Theoretic Decomposition Module for Multimodal Learning

This module provides information decomposition based on Partial Information 
Decomposition (PID) framework. Can be used independently or integrated with 
existing multimodal fusion models.

Usage:
    # Standalone pretraining
    from info_decomposition import pretrain_decomposition, InfoDecompositionModel
    
    model = InfoDecompositionModel(modality_dims=[100, 200, 300], 
                                   hidden_dim=128, num_classes=3)
    model = pretrain_decomposition(model, train_loader, args)
    
    # Integration with existing pipeline
    decomposed_features = model.decompose_features(modality_features)
    # Use decomposed_features['unique_features'], etc. in your fusion model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional


# ============== Core Decomposition Modules ==============

class ModalityDecompositionBranch(nn.Module):
    """
    Decomposes a single modality into information components:
    - Unique (U): Modality-specific information
    - Redundant (R): Information shared with other modalities  
    - Synergistic (S): Information emergent from combinations
    """
    
    def __init__(self, in_dim: int, out_dim: int, num_other_modalities: int,
                 dropout: float = 0.3, use_batchnorm: bool = True):
        super().__init__()
        self.num_other_modalities = num_other_modalities
        self.use_batchnorm = use_batchnorm
        
        # Unique information encoder
        layers_unique = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        ]
        self.fc_unique = nn.Sequential(*layers_unique)
        
        # Redundant information encoders (one per other modality)
        self.fc_redundant = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim, out_dim)
            ) for _ in range(num_other_modalities)
        ])
        
        # Synergistic information encoder
        layers_synergy = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        ]
        self.fc_synergy = nn.Sequential(*layers_synergy)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: [batch, (seq), in_dim]
        
        Returns:
            unique: [batch, (seq), out_dim]
            redundant: List of [batch, (seq), out_dim] tensors
            synergy: [batch, (seq), out_dim]
        """
        # Handle batch normalization for 3D inputs
        original_shape = x.shape
        if len(x.shape) == 3 and self.use_batchnorm:
            batch, seq, dim = x.shape
            x_flat = x.reshape(batch * seq, dim)
            
            unique = self.fc_unique(x_flat).reshape(batch, seq, -1)
            redundant = [fc(x_flat).reshape(batch, seq, -1) 
                        for fc in self.fc_redundant]
            synergy = self.fc_synergy(x_flat).reshape(batch, seq, -1)
        else:
            unique = self.fc_unique(x)
            redundant = [fc(x) for fc in self.fc_redundant]
            synergy = self.fc_synergy(x)
        
        return unique, redundant, synergy


class InfoDecompositionModel(nn.Module):
    """
    Multi-modal information decomposition model using PID framework.
    
    Decomposes N modalities into:
    - U_i: Unique information for modality i
    - R_{ij}: Redundant information between modalities i and j
    - S: Synergistic information from all modalities
    """
    
    def __init__(self, 
                 modality_dims: List[int],
                 hidden_dim: int,
                 num_classes: int,
                 dropout: float = 0.3,
                 use_batchnorm: bool = True):
        super().__init__()
        
        self.num_modalities = len(modality_dims)
        self.hidden_dim = hidden_dim
        
        # Per-modality encoders (optional preprocessing)
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) if dim != hidden_dim else nn.Identity()
            for dim in modality_dims
        ])
        
        # Decomposition branches
        self.decomposition_branches = nn.ModuleList([
            ModalityDecompositionBranch(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_other_modalities=self.num_modalities - 1,
                dropout=dropout,
                use_batchnorm=use_batchnorm
            ) for _ in range(self.num_modalities)
        ])
        
        # Prediction heads for each component
        self.unique_predictors = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) 
            for _ in range(self.num_modalities)
        ])
        self.redundant_predictor = nn.Linear(hidden_dim, num_classes)
        self.synergy_predictor = nn.Linear(hidden_dim, num_classes)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, modality_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            modality_features: List of [batch, (seq), dim] tensors
            
        Returns:
            Dictionary with decomposed representations and predictions
        """
        # Encode modalities to common dimension
        encoded = [encoder(feat) for encoder, feat in 
                  zip(self.modality_encoders, modality_features)]
        
        # Decompose each modality
        decompositions = [branch(feat) for branch, feat in 
                         zip(self.decomposition_branches, encoded)]
        
        # Extract components
        uniques = [d[0] for d in decompositions]
        redundants = [d[1] for d in decompositions]  # List of lists
        synergies = [d[2] for d in decompositions]
        
        # Aggregate redundant information
        redundant_agg = self._aggregate_redundant(redundants)
        
        # Aggregate synergistic information
        synergy_agg = torch.stack(synergies).mean(dim=0)
        
        # Aggregate unique information
        unique_agg = torch.stack(uniques).mean(dim=0)
        
        # Combined representation
        combined = torch.cat([unique_agg, redundant_agg, synergy_agg], dim=-1)
        final_logits = self.fusion(combined)
        
        return {
            'unique_features': uniques,
            'redundant_features': redundant_agg,
            'synergy_features': synergy_agg,
            'unique_agg': unique_agg,
            'combined': combined,
            'logits': final_logits,
            'decompositions': decompositions  # Raw decompositions
        }
    
    def _aggregate_redundant(self, redundants: List[List[torch.Tensor]]) -> torch.Tensor:
        """Aggregate redundant representations across modality pairs"""
        all_redundant = []
        for i in range(self.num_modalities):
            for j, red_list in enumerate(redundants):
                if i != j:
                    idx = i if i < j else i - 1
                    if idx < len(red_list):
                        all_redundant.append(red_list[idx])
        
        if len(all_redundant) == 0:
            # Fallback if no redundant features
            return torch.zeros_like(redundants[0][0]) if len(redundants) > 0 and len(redundants[0]) > 0 else torch.zeros(1, 1, self.hidden_dim)
        
        return torch.stack(all_redundant).mean(dim=0)
    
    def decompose_features(self, modality_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convenience method to extract decomposed features without training.
        Useful for integration with existing models.
        """
        with torch.no_grad():
            return self.forward(modality_features)


# ============== Information-Theoretic Loss Functions ==============

def mutual_information_estimate(z1: torch.Tensor, z2: torch.Tensor, 
                               method: str = 'correlation') -> torch.Tensor:
    """
    Estimate mutual information I(Z1; Z2) between two representations.
    
    Args:
        z1, z2: [batch, (seq), dim] tensors
        method: 'correlation' (differentiable) or 'kl' (more accurate)
    
    Returns:
        Scalar MI estimate
    """
    # Flatten if sequential
    z1_flat = z1.reshape(-1, z1.shape[-1])
    z2_flat = z2.reshape(-1, z2.shape[-1])
    
    if method == 'correlation':
        # Correlation-based approximation (differentiable)
        z1_centered = z1_flat - z1_flat.mean(dim=0, keepdim=True)
        z2_centered = z2_flat - z2_flat.mean(dim=0, keepdim=True)
        
        cov_matrix = torch.mm(z1_centered.t(), z2_centered) / z1_flat.shape[0]
        mi_proxy = torch.norm(cov_matrix, p='fro') ** 2
        
    elif method == 'kl':
        # KL-based estimate (less differentiable but more accurate)
        # Using MINE (Mutual Information Neural Estimation) approximation
        mi_proxy = compute_mine_lower_bound(z1_flat, z2_flat)
    
    return mi_proxy


def compute_mine_lower_bound(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """MINE lower bound on mutual information (simplified)"""
    # This is a placeholder - full MINE requires a critic network
    # For now, use correlation-based proxy
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    cov = torch.mm(z1_centered.t(), z2_centered) / z1.shape[0]
    return torch.norm(cov, p='fro') ** 2


def uniqueness_loss(unique: torch.Tensor,
                   redundants: List[torch.Tensor],
                   synergy: torch.Tensor,
                   alpha: float = 1.0) -> torch.Tensor:
    """
    Encourage unique representation to be independent of redundant and synergistic parts.
    Minimize I(U; R_i) and I(U; S)
    """
    loss = 0.0
    
    # U independent of all R's
    for red in redundants:
        loss += mutual_information_estimate(unique, red)
    
    # U independent of S
    loss += mutual_information_estimate(unique, synergy)
    
    return alpha * loss / (len(redundants) + 1)


def redundancy_loss(redundants_per_modality: List[List[torch.Tensor]],
                   beta: float = 1.0) -> torch.Tensor:
    """
    Encourage redundant parts to be correlated across modalities.
    Maximize I(R_i; R_j) for corresponding redundant pairs
    """
    loss = 0.0
    count = 0
    
    for i in range(len(redundants_per_modality)):
        for j in range(i + 1, len(redundants_per_modality)):
            reps_i = redundants_per_modality[i]
            reps_j = redundants_per_modality[j]
            
            # Match corresponding redundant components
            for k in range(min(len(reps_i), len(reps_j))):
                # Negative MI = maximize correlation
                loss -= mutual_information_estimate(reps_i[k], reps_j[k])
                count += 1
    
    return beta * loss / max(count, 1)


def synergy_loss(synergies: List[torch.Tensor],
                uniques: List[torch.Tensor],
                redundant_agg: torch.Tensor,
                gamma: float = 1.0) -> torch.Tensor:
    """
    Encourage synergistic representation to capture emergent information.
    Minimize I(S; U_i) + I(S; R) to ensure synergy is distinct
    """
    synergy_agg = torch.stack(synergies).mean(dim=0)
    
    loss = 0.0
    
    # S independent of each U
    for u in uniques:
        loss += mutual_information_estimate(synergy_agg, u)
    
    # S independent of R
    loss += mutual_information_estimate(synergy_agg, redundant_agg)
    
    return gamma * loss / (len(uniques) + 1)


def compute_pid_loss(outputs: Dict[str, torch.Tensor],
                    labels: torch.Tensor,
                    alpha_u: float = 1.0,
                    alpha_r: float = 1.0,
                    alpha_s: float = 0.5,
                    lambda_pred: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Complete PID-based loss function.
    
    Args:
        outputs: Model outputs from forward pass
        labels: Ground truth labels
        alpha_*: Weights for different loss components
        lambda_pred: Weight for prediction loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Individual loss components
    """
    uniques = outputs['unique_features']
    redundants = [outputs['decompositions'][i][1] for i in range(len(outputs['decompositions']))]
    synergies = outputs['synergy_features']
    redundant_agg = outputs['redundant_features']
    logits = outputs['logits']
    
    # Flatten logits and labels if sequential
    if logits.dim() == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    if labels.dim() > 1:
        labels = labels.reshape(-1)
    
    # 1. Prediction loss
    criterion = nn.CrossEntropyLoss()
    L_pred = criterion(logits, labels)
    
    # 2. Uniqueness loss (minimize correlation with R and S)
    L_unique = sum([
        uniqueness_loss(u, r, s, alpha=1.0)
        for u, r, s in zip(uniques, redundants, [synergies] * len(uniques))
    ]) / len(uniques)
    
    # 3. Redundancy loss (maximize correlation between redundant parts)
    L_redundant = redundancy_loss(redundants, beta=1.0)
    
    # 4. Synergy loss (ensure synergy is distinct)
    L_synergy = synergy_loss([synergies], uniques, redundant_agg, gamma=1.0)
    
    # Total loss
    total_loss = (lambda_pred * L_pred + 
                 alpha_u * L_unique + 
                 alpha_r * L_redundant + 
                 alpha_s * L_synergy)
    
    loss_dict = {
        'total': total_loss.item(),
        'pred': L_pred.item(),
        'unique': L_unique.item() if isinstance(L_unique, torch.Tensor) else L_unique,
        'redundant': L_redundant.item() if isinstance(L_redundant, torch.Tensor) else L_redundant,
        'synergy': L_synergy.item() if isinstance(L_synergy, torch.Tensor) else L_synergy,
    }
    
    return total_loss, loss_dict


# ============== Training Functions ==============

def pretrain_decomposition(model: InfoDecompositionModel,
                          train_loader,
                          args,
                          n_epochs: int = 200,
                          device: str = 'cuda',
                          verbose: bool = True) -> InfoDecompositionModel:
    """
    Pretrain the information decomposition model.
    
    Args:
        model: InfoDecompositionModel instance
        train_loader: DataLoader with modality features
        args: Training arguments (must have lr, weight_decay, grad_norm_max)
        n_epochs: Number of pretraining epochs
        device: Device to train on
        verbose: Print training progress
        
    Returns:
        Trained model
    """
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(args, 'lr', 1e-4),
        weight_decay=getattr(args, 'weight_decay', 1e-5)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    if verbose:
        print(f"\n{'='*80}")
        print("Pretraining Information Decomposition Model")
        print(f"{'='*80}\n")
    
    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_loss_dict = {'pred': [], 'unique': [], 'redundant': [], 'synergy': []}
        
        for batch_idx, batch in enumerate(train_loader):
            # Extract modality features (adjust based on your data format)
            modality_features = []
            for key in sorted(batch.get('tensor', {}).keys()):
                feat = batch['tensor'][key].to(device)
                # Handle different tensor shapes
                if feat.dim() == 3:
                    feat = feat.permute(1, 2, 0).transpose(1, 2)
                modality_features.append(feat)
            
            labels = batch.get('labels', batch.get('label_tensor')).to(device)
            
            # Forward pass
            outputs = model(modality_features)
            
            # Compute PID loss
            alpha_u = getattr(args, 'decomposition_alpha_u', 1.0)
            alpha_r = getattr(args, 'decomposition_alpha_r', 1.0)
            alpha_s = getattr(args, 'decomposition_alpha_s', 0.5)
            
            loss, loss_dict = compute_pid_loss(
                outputs, labels, 
                alpha_u=alpha_u, alpha_r=alpha_r, alpha_s=alpha_s
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                getattr(args, 'grad_norm_max', 1.0)
            )
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            for key in epoch_loss_dict.keys():
                epoch_loss_dict[key].append(loss_dict[key])
            
            if verbose and batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss_dict['total']:.4f}, "
                      f"Pred: {loss_dict['pred']:.4f}, "
                      f"U: {loss_dict['unique']:.4f}, "
                      f"R: {loss_dict['redundant']:.4f}, "
                      f"S: {loss_dict['synergy']:.4f}")
        
        scheduler.step()
        
        if verbose:
            avg_loss = np.mean(epoch_losses)
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Pred: {np.mean(epoch_loss_dict['pred']):.4f}")
            print(f"  Unique: {np.mean(epoch_loss_dict['unique']):.4f}")
            print(f"  Redundant: {np.mean(epoch_loss_dict['redundant']):.4f}")
            print(f"  Synergy: {np.mean(epoch_loss_dict['synergy']):.4f}\n")
    
    if verbose:
        print("Pretraining complete!\n")
    
    model._pretrained = True
    return model


# ============== Utility Functions ==============

def load_pretrained_decomposition(checkpoint_path: str,
                                 modality_dims: List[int],
                                 hidden_dim: int,
                                 num_classes: int,
                                 device: str = 'cuda') -> InfoDecompositionModel:
    """Load a pretrained decomposition model from checkpoint"""
    model = InfoDecompositionModel(
        modality_dims=modality_dims,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model._pretrained = True
    model.to(device)
    model.eval()
    return model


def visualize_decomposition(model: InfoDecompositionModel,
                           modality_features: List[torch.Tensor],
                           method: str = 'pca'):
    """
    Visualize decomposed representations using PCA or t-SNE.
    Requires sklearn.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn and matplotlib required for visualization")
        return
    
    model.eval()
    with torch.no_grad():
        outputs = model.decompose_features(modality_features)
    
    # Extract components
    unique_agg = outputs['unique_agg'].cpu().numpy()
    redundant = outputs['redundant_features'].cpu().numpy()
    synergy = outputs['synergy_features'].cpu().numpy()
    
    # Flatten if sequential
    if unique_agg.ndim == 3:
        unique_agg = unique_agg.reshape(-1, unique_agg.shape[-1])
        redundant = redundant.reshape(-1, redundant.shape[-1])
        synergy = synergy.reshape(-1, synergy.shape[-1])
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42)
    
    unique_2d = reducer.fit_transform(unique_agg)
    redundant_2d = reducer.fit_transform(redundant)
    synergy_2d = reducer.fit_transform(synergy)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].scatter(unique_2d[:, 0], unique_2d[:, 1], alpha=0.5)
    axes[0].set_title('Unique Information')
    
    axes[1].scatter(redundant_2d[:, 0], redundant_2d[:, 1], alpha=0.5)
    axes[1].set_title('Redundant Information')
    
    axes[2].scatter(synergy_2d[:, 0], synergy_2d[:, 1], alpha=0.5)
    axes[2].set_title('Synergistic Information')
    
    plt.tight_layout()
    plt.savefig(f'decomposition_viz_{method}.png')
    print(f"Visualization saved to decomposition_viz_{method}.png")