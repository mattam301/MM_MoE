#!/bin/bash
set -e

# ============================================================================
# CONFIG
# ============================================================================

export DEVICE=0
export DATA=mosi
export MODALITY=TVA
export SCRIPT=src/imoe/train_interpretcc_new.py

COMMON_ARGS="\
--data ${DATA} \
--modality ${MODALITY} \
--device ${DEVICE} \
--n_runs 1 \
--num_workers 4 \
--pin_memory True \
--use_common_ids True \
--save True \
--debug False \
"

echo "========================================================================"
echo "Unified iMoE Experimental Sweep"
echo "Device: ${DEVICE} | Dataset: ${DATA} | Modality: ${MODALITY}"
echo "========================================================================"

# # ============================================================================
# # 1️⃣ SMOKE TEST (FAST, SMALL, SHOULD NEVER FAIL)
# # ============================================================================

# echo ""
# echo "==================== [1] SMOKE TEST ===================="

# CUDA_VISIBLE_DEVICES=${DEVICE} python ${SCRIPT} \
# ${COMMON_ARGS} \
# --seed 0 \
# --train_epochs 3 \
# --batch_size 4 \
# --lr 1e-4 \
# --hidden_dim 64 \
# --fusion_sparse False \
# --use_info_decomposition False

# # ============================================================================
# # 2️⃣ BATCH SIZE STRESS TEST (CATCHES COLLAPSE BUGS)
# # ============================================================================

# echo ""
# echo "==================== [2] BATCH SIZE STRESS ===================="

# for BS in 1 2 4 8 16 32; do
#   echo "---- Batch Size = ${BS} ----"
#   CUDA_VISIBLE_DEVICES=${DEVICE} python ${SCRIPT} \
#   ${COMMON_ARGS} \
#   --seed 0 \
#   --train_epochs 5 \
#   --batch_size ${BS} \
#   --lr 1e-4 \
#   --hidden_dim 128 \
#   --fusion_sparse False \
#   --use_info_decomposition False
# done

# ============================================================================
# 3️⃣ DECOMPOSITION ABLATION
# ============================================================================

# echo ""
# echo "==================== [3] DECOMPOSITION ABLATION ===================="
# for i in 1 2 3 4 5 6 7 8; do
#   for DECOMP in False True; do
#     echo "---- use_enhanced_pid = ${DECOMP} ----"
#     CUDA_VISIBLE_DEVICES=${DEVICE} python ${SCRIPT} \
#     ${COMMON_ARGS} \
#     --seed $i \
#     --train_epochs 30 \
#     --batch_size 32 \
#     --lr 1e-4 \
#     --hidden_dim 128 \
#     --fusion_sparse False \
#     --use_info_decomposition ${DECOMP} \
#     --use_enhanced_pid ${DECOMP} \
#     --decomposition_loss_weight 0.01 \
#     --use_comet True
#   done
# done
# # ============================================================================
# # 4️⃣ HYPERPARAMETER SWEEPS (CORE ONES THAT MATTER)
# # ============================================================================

# echo ""
# echo "==================== [4] HYPERPARAMETER SWEEP ===================="

# for LR in 1e-3 5e-4 1e-4; do
#   for IW in 0.0 0.05 0.1; do
#     for TEMP in 0.5 1.0 2.0; do
#       echo "---- LR=${LR} | Interaction=${IW} | Temp=${TEMP} ----"
#       CUDA_VISIBLE_DEVICES=${DEVICE} python ${SCRIPT} \
#       ${COMMON_ARGS} \
#       --seed 0 \
#       --train_epochs 15 \
#       --batch_size 32 \
#       --lr ${LR} \
#       --interaction_loss_weight ${IW} \
#       --temperature_rw ${TEMP} \
#       --hidden_dim 128 \
#       --fusion_sparse False \
#       --use_info_decomposition True \
#       --decomposition_loss_weight 0.01
#     done
#   done
# done

# # ============================================================================
# # 5️⃣ FULL TRAIN (REFERENCE RUN)
# # ============================================================================

echo ""
echo "==================== [5] FULL TRAIN (REFERENCE) ===================="

CUDA_VISIBLE_DEVICES=${DEVICE} python ${SCRIPT} \
${COMMON_ARGS} \
--seed 0 \
--train_epochs 50 \
--batch_size 32 \
--lr 1e-4 \
--weight_decay 1e-5 \
--hidden_dim 128 \
--hidden_dim_rw 256 \
--num_layer_rw 2 \
--temperature_rw 1.0 \
--interaction_loss_weight 0.1 \
--fusion_sparse False \
--use_info_decomposition True \
--use_enhanced_pid True \
--decomposition_loss_weight 0.01 \
--use_comet True

# echo ""
# echo "========================================================================"
# echo "✓ All experiments completed successfully"
# echo "========================================================================"