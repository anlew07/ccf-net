# Stage 2 v1 (minimal runnable)

Files:
- `stage2_dataset.py`: loads image + Stage1Label + TrustMap + original Scribble
- `stage2_loss.py`: trust-weighted pseudo supervision + scribble hard anchor + light smoothness
- `stage2_train.py`: training entry using `dino1_net.py`
- `stage2_test.py`: test/eval entry using the trained checkpoint

Recommended training command:

```bash
python stage2_train.py \
  --data_root CodDataset \
  --stage1_root runs/stage1/export_mainprob \
  --snapshot /root/shared-nvme/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations/cp_cof_1_1_dino_sem_edge/dino_0415/model-best.pth \
  --save_root ./out_stage2 \
  --exp_name stage2_mainprob_v1 \
  --batch_size 16 \
  --epochs 150
```

Recommended test command:

```bash
python stage2_test.py \
  --ckpt ./out_stage2/stage2_mainprob_v1/model-best.pth \
  --exp_name stage2_mainprob_v1_eval
```
