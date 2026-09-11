# Revision P0 Experiments

This directory is reserved for first-paper revision experiments only.

## Ground rules

- `main` is not modified by revision experiments.
- All model variants must be derived from the current canonical `ccfnet.py` on this branch.
- Dataset layout, DINOv3 checkpoint path, training losses, image size, evaluation toolkit, and other server-side paths remain consistent with the formal project unless an experiment explicitly changes one factor.
- Do not reuse historical ablation numbers unless the corresponding architecture is verified to match the current canonical model.
- Before any long training, run the sanity checks below.

## Phase 0: sanity checks

1. Verify the existing best checkpoint with the current `ccfnet.py` and `test.py`.
2. Run `sanity/check_runtime.py` to record the actual server environment and required paths.
3. Run `sanity/grad_check.py` for exactly one mini-batch. It performs forward/backward only and **does not call `optimizer.step()`** or save a checkpoint. The mini-batch size is 2 (rather than 1) because the current PyramidPooling module contains BatchNorm after 1x1 adaptive pooling, which is invalid in training mode with batch size 1.
4. Run a short reproduction only after the gradient/computation-graph result is understood.
5. Run the full 150-epoch reproduction only after the short run is healthy.

## Planned P0 experiment chain

- R0: full-model reproduction using the canonical training path.
- R1: matched DINOv3 baseline, disabling CP and TFGM while keeping the current decoder unchanged.
- R2: TFGM without CP discrepancy guidance.
- R3: full CP-guided TFGM (canonical model).
- A1: CP cue vs. prediction-error correlation analysis, no retraining.
- A2: comparison with alternative uncertainty/disagreement cues, no retraining where possible.
- A3: spectral analysis before/after TFGM, no retraining.

Additional semantic-only, edge-only, stop-gradient, fusion-weight, and sparsification experiments will be added only after the R0-R3 chain is validated.

## Server assumptions inherited from the project

- DINOv3 source: `/root/shared-nvme/dinov3-main`
- DINOv3 checkpoint: `/root/shared-nvme/pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- Dataset root: `CodDataset`
- Scribble labels: `CodDataset/train/Scribble`
- Test datasets: CAMO, CHAMELEON, COD10K, NC4K under `CodDataset/test/`
- Existing best checkpoint expected by `test.py`: `cp_cof_1_1_dino_sem_edge/dino_0415/model-best.pth`

## Important local-worktree note

When `revision-p0-experiments` was created, the local Mac worktree already showed uncommitted modifications in `ccfnet.py`, `train.py`, and `test.py`. Those local changes were **not** part of the remote branch at creation time. The sanity scripts here are intentionally additive and do not overwrite those files. Preserve and review those local diffs separately before deciding whether they belong in the formal reproduction path.
