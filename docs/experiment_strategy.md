# Experiment strategy: iMoE perturbations + PID

## Goal

Evaluate whether replacing the legacy “random vector” modality perturbation with **on-manifold** replacements (mean / zero / PID-derived) improves:

- Task performance (Acc/F1/AUC or MAE/Corr)
- Expert specialization signals (interaction losses)
- Robustness / stability (no collapse, consistent training curves)

## Key switches

- `--use_info_decomposition {True|False}`
  - Enables the Enhanced PID module and adds decomposition regularization.
- `--perturbation_mode {random|zero|mean_batch|pid_drop_unique}`
  - Controls how each modality is replaced during the perturbed forward passes (used for interaction loss).
- `--num_perturb N`
  - If `N>0`, only perturb **N modalities** per iteration (legacy “less perturbed” ablation).

## Recommended experiment ladder

### Phase 0: sanity baselines (should always run)

- **Baseline A (legacy)**:
  - `--use_info_decomposition False`
  - `--perturbation_mode random`
  - `--num_perturb 0`

- **Baseline B (mean)**:
  - `--use_info_decomposition False`
  - `--perturbation_mode mean_batch`

- **Baseline C (zero)**:
  - `--use_info_decomposition False`
  - `--perturbation_mode zero`

### Phase 1: PID regularization without changing perturbations

- `--use_info_decomposition True`
- `--perturbation_mode random`
- Sweep `--decomposition_loss_weight` in `{0.0, 0.01, 0.02, 0.05}`

### Phase 2: PID-derived perturbations (your core idea)

- `--use_info_decomposition True`
- `--perturbation_mode pid_drop_unique`
  - Replacement per modality \(i\): reconstruct using **redundant + synergy** (drop unique).
- Sweep:
  - `--decomposition_loss_weight` in `{0.01, 0.02, 0.05}`
  - `--interaction_loss_weight` in `{0.05, 0.1, 0.2}`

### Phase 3: less-perturbed variant

Repeat Phase 0/2 with:

- `--num_perturb` in `{1, 2}` (for 3 modalities)

## What to report (minimum)

- **Performance**: Acc/F1/AUC (classification) or MAE/Corr (regression)
- **Training stability**: loss curves, any NaNs, variance across seeds
- **Specialization proxies**: per-component interaction losses (uni/syn/red)
- **PID diagnostics** (if enabled): component weights and sample-level PID tracker outputs

## Notes

- `pid_drop_unique` requires `--use_info_decomposition True` (by design).
- `mean_batch` is a strong “on-manifold but low-information” baseline; it’s the closest legacy ablation to PID perturbation in spirit.

