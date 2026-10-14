# DAiSEE fine-tuning with VideoMAE v2

`main_pipeline.py` fine-tunes the existing VideoMAE v2 model implementation
for four-level student **Engagement** recognition on DAiSEE. It uses the
official DAiSEE Train/Validation/Test split discovered under the dataset root;
it never makes a random split.

## Before running

Run these commands in the VideoMAEv2 repository directory, in the existing
VideoMAE v2 Python environment. No package installation is required.

```bash
cd /path/to/VideoMAEv2
python --version
python -c "import torch, timm, decord; print(torch.__version__, torch.cuda.is_available())"
```

On the target WSL/Windows machine, the DAiSEE root is expected at:

```text
/mnt/c/Users/puneet/Downloads/DAiSEE
```

The script automatically looks for the normal published layout, such as:

```text
DAiSEE/
├── DataSet/Train/
├── DataSet/Validation/
├── DataSet/Test/
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

Video files may be nested below each split directory. Label CSVs must contain
`ClipID` and `Engagement` columns. DAiSEE engagement values must be integers
from 0 through 3.

## Train

Use a VideoMAE v2 pretrained checkpoint. The repository's fine-tuning scripts
refer to a suitable base checkpoint as `vit_b_hybrid_pt_800e.pth`. That is a
local filename convention: the current official base pretrained weight is
published as `mae-b/pytorch_model.bin`.

### Download the pretrained base checkpoint

From the VideoMAEv2 directory on the GPU machine, download it once and give it
the local filename used by the command below:

```bash
mkdir -p checkpoints
wget -O checkpoints/vit_b_hybrid_pt_800e.pth \
  https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/mae-b/pytorch_model.bin
```

If `wget` is unavailable, use `curl` instead:

```bash
mkdir -p checkpoints
curl -L --fail --output checkpoints/vit_b_hybrid_pt_800e.pth \
  https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/mae-b/pytorch_model.bin
```

Confirm the exact absolute path with:

```bash
realpath checkpoints/vit_b_hybrid_pt_800e.pth
```

Pass that output to `--pretrained`. The training script loads compatible
VideoMAE-v2 backbone parameters and deliberately leaves DAiSEE's new 4-class
classification head randomly initialized.

```bash
python main_pipeline.py \
  --data-path /mnt/c/Users/puneet/Downloads/DAiSEE \
  --pretrained /mnt/c/Users/puneet/Jeet/VideoMAEv2/checkpoints/vit_b_hybrid_pt_800e.pth\
  --output-dir ./outputs/daisee_vitb \
  --batch-size 4 \
  --grad-accum 4
```

The default clip is 16 frames sampled every 4 decoded frames, at 224×224.
With batch size 4 and gradient accumulation 4, the effective single-process
batch size is 16 clips. Reduce `--batch-size` and increase `--grad-accum` if
GPU memory is insufficient.

Useful changes:

```bash
# A shorter temporal extent / less VRAM
--num-frames 8 --sampling-rate 4

# More temporal coverage, if memory permits
--num-frames 32 --sampling-rate 2

# CPU-only smoke run (very slow for a real experiment)
--device cpu --batch-size 1 --num-workers 0 --no-amp
```

For multi-GPU training, launch through `torchrun`:

```bash
torchrun --nproc_per_node=2 main_pipeline.py \
  --data-path /mnt/c/Users/puneet/Downloads/DAiSEE \
  --pretrained /path/to/vit_b_hybrid_pt_800e.pth \
  --output-dir ./outputs/daisee_vitb_2gpu \
  --batch-size 4 --grad-accum 2
```

## Evaluate the best checkpoint

```bash
python main_pipeline.py \
  --data-path /mnt/c/Users/puneet/Downloads/DAiSEE \
  --output-dir ./outputs/daisee_vitb \
  --resume ./outputs/daisee_vitb/best.pth \
  --eval-only
```

Evaluation averages the default three deterministic temporal clips for each
test video. Use `--test-clips 1` for a faster single-clip test.

## Dry run (recommended before a long job)

This performs the complete split/leakage/video-decode audit, loads the
checkpoint, and runs one training batch through the model. It prints the input,
label, and logits shapes, then exits without updating weights or saving a
checkpoint.

```bash
python main_pipeline.py \
  --data-path /mnt/c/Users/puneet/Downloads/DAiSEE \
  --pretrained ./checkpoints/vit_b_hybrid_pt_800e.pth \
  --output-dir ./outputs/daisee_dry_run \
  --batch-size 1 --num-workers 0 \
  --dry-run
```

## Outputs

The output directory contains:

- `data_audit.json` — detected split/label sources and leakage-audit counts.
- `metrics.jsonl` — train/validation loss, accuracy, macro-F1, MAE, QWK, and
  per-class metrics after each epoch.
- `best.pth` — checkpoint selected by **validation macro-F1**.
- `last.pth` — latest checkpoint for resuming.
- `test_metrics.json` — final test accuracy, macro-F1, per-class
  precision/recall/F1, MAE, QWK, and confusion matrix.

## Important safety checks

Before the first epoch, the script checks that every official split is present,
labels resolve to exactly one video, labels are in range, and ClipIDs, physical
paths, and inferred participant IDs do not overlap across splits. It also
decodes each labelled video once by default. A failed check stops training;
the script never silently replaces a bad sample with another video. A labelled
video that is physically missing is the one exception: it is excluded, printed
as a warning, and listed in `data_audit.json` so its omission is explicit.

The default participant rule uses the first six digits of a ClipID. If your
DAiSEE release uses another identity convention, supply a regex whose first
capture group is the participant ID:

```bash
--subject-regex '^([A-Za-z0-9_-]+)_' 
```

Only use `--skip-video-verification` after a successful full preflight. Use
`--full-duplicate-audit` when you want the stronger (but slower) SHA-1 audit
of all labelled videos.
