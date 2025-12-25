#!/bin/bash

# ============================================================================
# QUICK SINGLE RUN FOR TESTING
# ============================================================================

export device=0

echo "========================================================================"
echo "Unified iMoE Training: Pre-train Decomposition + Train Experts"
echo "========================================================================"

CUDA_VISIBLE_DEVICES=$device python src/imoe/train_interpretcc_new.py \
--data mosi \
--modality TVA \
--device $device \
--seed 0 \
--n_runs 1 \
--num_workers 4 \
--pin_memory True \
--use_common_ids True \
--save True \
--debug False \
\
`# ========== PHASE 1: PRE-TRAINING DECOMPOSITION (10 epochs) ==========` \
--use_pretrain True \
--pretrain_epochs 20 \
--pretrain_lr 1e-3 \
--pretrain_method reconstruction \
--pretrain_weight_decay 1e-5 \
\
`# ========== PHASE 2: MAIN TRAINING (50 epochs) ==========` \
--train_epochs 50 \
--batch_size 32 \
--lr 1e-4 \
--weight_decay 1e-5 \
--freeze_decomposer False \
\
`# ========== REWEIGHTING MODEL ==========` \
--temperature_rw 1.0 \
--hidden_dim_rw 256 \
--num_layer_rw 2 \
\
`# ========== LOSS WEIGHTS ==========` \
--interaction_loss_weight 0.1 \
--orthogonality_loss_weight 1e-3 \
--load_balancing_weight 1e-2 \
\
`# ========== ARCHITECTURE ==========` \
--fusion_sparse False \
--hidden_dim 128 \
--num_layers_enc 2 \
--num_layers_fus 2 \
--num_layers_pred 2 \
\
`# ========== GUMBEL-SOFTMAX ==========` \
--tau 1.0 \
--hard True \
--threshold 0.6 \
--dropout 0.5 \
\
`# ========== DECOMPOSITION: 2 Experts per Type (U/R/S) ==========` \
--use_decomposition True \
--decomposition_method learned \
--num_experts_per_type 2 \
--expert_specialization True \
\
`# ========== INTERPRETABILITY ==========` \
--log_interactions True \
--save_interaction_maps False

echo ""
echo "========================================================================"
echo "✓ Training completed!"
echo "========================================================================"