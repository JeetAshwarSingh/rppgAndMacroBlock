#!/usr/bin/env python3
"""Fine-tune the repository's VideoMAE v2 models on DAiSEE engagement.

DAiSEE supplies four *affect dimensions* (Boredom, Engagement, Confusion,
Frustration), each annotated on the ordered 0--3 scale.  This script targets
the Engagement column by default.  Its four values are trained as a 4-way
softmax classifier, the established directly comparable DAiSEE protocol.
They are ordinal, however, so MAE and quadratic weighted kappa are reported
as secondary diagnostics; they must not be confused with the selection metric
(validation macro-F1, which is robust to DAiSEE class imbalance).

The script deliberately does not use dataset.VideoClsDataset: that class
randomly substitutes another sample after a decode failure, which invalidates
the label/sample accounting needed for a small affect dataset.  It reuses this
repository's Decord loader, video transforms, VideoMAE-v2 model registry and
model implementation without copying the architecture.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from timm.models import create_model

# Registers the VideoMAE-v2 model names with timm. Do not remove.
import models  # noqa: F401
from dataset import video_transforms, volume_transforms
from dataset.datasets import spatial_sampling, tensor_normalize
from dataset.loader import get_video_loader
from dataset.random_erasing import RandomErasing

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
VIDEO_SUFFIXES = {'.avi', '.mp4', '.mov', '.mkv', '.webm', '.mpeg', '.mpg'}
SPLIT_ALIASES = {
    'train': ('train', 'training'),
    'val': ('validation', 'valid', 'val', 'dev'),
    'test': ('test', 'testing'),
}


def norm_name(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def setup_distributed(args):
    args.distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if not args.distributed:
        args.rank, args.world_size, args.local_rank = 0, 1, 0
        return
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.device.startswith('cuda'):
        torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')


def is_main(args):
    return args.rank == 0


def seed_everything(seed, rank):
    seed += rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_split_dir(root, split):
    """Find an official split directory, never synthesize a split."""
    aliases = {norm_name(x) for x in SPLIT_ALIASES[split]}
    candidates = []
    for p in root.rglob('*'):
        if p.is_dir() and norm_name(p.name) in aliases:
            candidates.append(p)
    # Prefer DataSet/Train (the published DAiSEE layout) to unrelated folders.
    candidates.sort(key=lambda p: (0 if norm_name(p.parent.name) in
                    {'dataset', 'data'} else 1, len(p.parts), str(p)))
    if not candidates:
        raise FileNotFoundError(f'Official DAiSEE {split} directory not found under {root}')
    return candidates[0]


def find_label_csv(root, split):
    aliases = [norm_name(x) for x in SPLIT_ALIASES[split]]
    candidates = []
    for p in root.rglob('*.csv'):
        name = norm_name(p.stem)
        if any(a in name for a in aliases) and 'label' in name:
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f'Official DAiSEE {split} labels CSV not found under {root}')
    return sorted(candidates, key=lambda p: (0 if norm_name(p.parent.name) == 'labels' else 1, str(p)))[0]


def canonical_id(value):
    return Path(str(value).strip()).stem.lower()


def read_labels(csv_path, target):
    """Return id->integer class, accepting normal DAiSEE CSV header variants."""
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    if not rows or not rows[0]:
        raise ValueError(f'Empty/unreadable labels file: {csv_path}')
    fields = {norm_name(k): k for k in rows[0]}
    id_key = next((fields[k] for k in ('clipid', 'clip', 'videoid', 'video', 'id') if k in fields), None)
    target_key = fields.get(norm_name(target))
    if id_key is None or target_key is None:
        raise ValueError(f'{csv_path} needs ClipID and {target}; headers are {list(rows[0])}')
    result = {}
    for line, row in enumerate(rows, start=2):
        key = canonical_id(row[id_key])
        try:
            label = int(float(row[target_key]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid {target} label at {csv_path}:{line}') from exc
        if key in result:
            raise ValueError(f'Duplicate ClipID {key} in {csv_path}')
        result[key] = label
    return result


def index_videos(split_dir):
    """Index only videos physically below this official split directory."""
    by_id = defaultdict(list)
    for path in split_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            by_id[canonical_id(path.name)].append(path.resolve())
    return by_id


def infer_subject(clip_id, path, subject_regex):
    """DAiSEE IDs normally start with the six-digit participant id.

    --subject-regex can override this if a local release names files differently.
    A failed inference is a hard error: a subject-leakage audit cannot be honest
    without a person identifier.
    """
    match = re.search(subject_regex, clip_id)
    if not match:
        raise ValueError(f'Cannot infer subject from {clip_id}; set --subject-regex')
    return match.group(1) if match.groups() else match.group(0)


def build_manifest(root, split, target, args):
    split_dir, csv_path = find_split_dir(root, split), find_label_csv(root, split)
    labels, videos = read_labels(csv_path, target), index_videos(split_dir)
    records, missing, ambiguous = [], [], []
    for clip_id, label in labels.items():
        matches = videos.get(clip_id, [])
        if len(matches) == 1:
            records.append({'id': clip_id, 'path': str(matches[0]), 'label': label,
                            'subject': infer_subject(clip_id, matches[0], args.subject_regex)})
        elif not matches:
            missing.append(clip_id)
        else:
            ambiguous.append((clip_id, [str(x) for x in matches]))
    extras = sorted(set(videos) - set(labels))
    # A missing file is a recoverable dataset-delivery issue.  Exclude only that
    # label/file pair, make the exclusion auditable, and never replace it with a
    # different sample.  Ambiguity is not recoverable because it can select the
    # wrong video for a label.
    if ambiguous:
        raise RuntimeError(f'{split}: {len(ambiguous)} IDs resolve to multiple videos. '
                           f'Examples ambiguous={ambiguous[:1]}')
    if not records:
        raise RuntimeError(f'{split}: no labelled videos resolved from {split_dir}')
    labels_seen = {r['label'] for r in records}
    invalid = sorted(labels_seen - set(range(args.num_classes)))
    if invalid:
        raise ValueError(f'{split}: labels {invalid} outside 0..{args.num_classes - 1}; do not silently remap.')
    if extras:
        warnings.warn(f'{split}: {len(extras)} unlabelled videos ignored (examples: {extras[:3]}).')
    return records, {'split_dir': str(split_dir), 'labels': str(csv_path),
                     'unlabelled_videos': len(extras), 'missing_labelled_videos': missing,
                     'num_missing_labelled_videos': len(missing)}


def small_fingerprint(path):
    """Streaming SHA-1 guard; opt-in because hashing every video is expensive."""
    stat = os.stat(path)
    digest = hashlib.sha1()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return (stat.st_size, digest.hexdigest())


def audit_manifests(manifests, args):
    ids = {k: {r['id'] for r in v} for k, v in manifests.items()}
    paths = {k: {os.path.realpath(r['path']) for r in v} for k, v in manifests.items()}
    subjects = {k: {r['subject'] for r in v} for k, v in manifests.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        for what, values in (('ClipID', ids), ('resolved video path', paths), ('subject', subjects)):
            overlap = values[left] & values[right]
            if overlap:
                raise RuntimeError(f'LEAKAGE: {len(overlap)} {what}s overlap {left}/{right}: {sorted(overlap)[:5]}')
    all_records = [r for split in manifests.values() for r in split]
    seen = {}
    for rec in all_records:
        fp = small_fingerprint(rec['path']) if args.full_duplicate_audit else None
        if fp is not None and fp in seen:
            raise RuntimeError(f'LEAKAGE: byte-identical videos: {rec["path"]} and {seen[fp]}')
        if fp is not None: seen[fp] = rec['path']
    return {'counts': {k: len(v) for k, v in manifests.items()},
            'subjects': {k: len(v) for k, v in subjects.items()},
            'class_counts_train': dict(sorted(Counter(r['label'] for r in manifests['train']).items()))}


def verify_videos(records, workers=1):
    """Decode metadata before training; failures are reported, never substituted."""
    loader = get_video_loader()
    bad = []
    for r in records:
        try:
            vr = loader(r['path'])
            if len(vr) < 1: raise RuntimeError('zero frames')
            _ = vr.get_batch([0]).asnumpy()
        except Exception as exc:  # Decoder errors are data-quality failures.
            bad.append({'id': r['id'], 'path': r['path'], 'error': str(exc)})
    if bad:
        raise RuntimeError(f'{len(bad)} corrupt/unreadable labelled videos; examples: {bad[:3]}')


class DAiSEEDataset(Dataset):
    def __init__(self, records, args, train=False, test_clips=1):
        self.records, self.args, self.train, self.test_clips = records, args, train, test_clips
        self.loader = get_video_loader()
        self.erase = RandomErasing(args.reprob, mode='pixel', max_count=1, device='cpu') if train and args.reprob else None
        self.eval_transform = video_transforms.Compose([
            video_transforms.Resize(args.short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop((args.input_size, args.input_size)),
            volume_transforms.ClipToTensor(), video_transforms.Normalize(MEAN, STD)])

    def __len__(self): return len(self.records) * self.test_clips

    def _indices(self, length, view):
        span = 1 + (self.args.num_frames - 1) * self.args.sampling_rate
        if length <= span:
            # linspace avoids repeated first frames and preserves increasing order.
            return np.linspace(0, max(length - 1, 0), self.args.num_frames).round().astype(np.int64)
        if self.train:
            start = np.random.randint(0, length - span + 1)
        elif self.test_clips == 1:
            start = (length - span) // 2
        else:
            start = round(view * (length - span) / (self.test_clips - 1))
        return start + np.arange(self.args.num_frames, dtype=np.int64) * self.args.sampling_rate

def _train_transform(self, frames):
        aug = video_transforms.create_random_augment((self.args.input_size, self.args.input_size),
                                                     self.args.aa, self.args.train_interpolation)
        frames = [transforms.ToPILImage()(x) for x in frames]
        frames = [transforms.ToTensor()(x) for x in aug(frames)]
        x = torch.stack(frames).permute(0, 2, 3, 1)  # T,H,W,C; frames are RGB
        x = tensor_normalize(x, MEAN, STD).permute(3, 0, 1, 2)
        x = spatial_sampling(x, spatial_idx=-1, min_scale=256, max_scale=320,
                             crop_size=self.args.input_size, random_horizontal_flip=True,
                             inverse_uniform_sampling=False, aspect_ratio=[0.75, 1.3333],
                             # DAiSEE is face-centric: action-recognition's 8% crops
                             # frequently remove the face and destabilize fine-tuning.
                             scale=[self.args.min_crop_scale, 1.0], motion_shift=False)
        if self.erase is not None:
            x = self.erase(x.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
        return x

    def __getitem__(self, index):
        record, view = self.records[index % len(self.records)], index // len(self.records)
        try:
            vr = self.loader(record['path'])
            if len(vr) < 1: raise RuntimeError('zero frames')
            frames = vr.get_batch(self._indices(len(vr), view).tolist()).asnumpy()  # Decord RGB: T,H,W,C
        except Exception as exc:
            # Do not resample another item: that silently changes class weights/metrics.
            raise RuntimeError(f'Decode failed for id={record["id"]}, path={record["path"]}: {exc}') from exc
        x = self._train_transform(frames) if self.train else self.eval_transform(frames)
        if tuple(x.shape[:2]) != (3, self.args.num_frames):
            raise RuntimeError(f'Bad tensor shape {tuple(x.shape)} for {record["path"]}; expected (3,T,H,W)')
        return x, record['label'], record['id']


def worker_init(seed):
    def init(worker_id):
        value = seed + worker_id
        random.seed(value); np.random.seed(value); torch.manual_seed(value)
    return init


def make_model(args):
    model = create_model(args.model, pretrained=False, img_size=args.input_size,
                         num_classes=args.num_classes, all_frames=args.num_frames,
                         tubelet_size=args.tubelet_size, drop_rate=args.drop,
                         drop_path_rate=args.drop_path, attn_drop_rate=0.,
                         head_drop_rate=args.head_drop, use_mean_pooling=True,
                         init_scale=args.init_scale, with_cp=args.with_checkpoint)
    if args.input_size % model.patch_embed.patch_size[0] or args.num_frames % args.tubelet_size:
        raise ValueError('input-size must divide patch size and num-frames must divide tubelet-size')
    return model


def load_weights(model, checkpoint_path, resume=False):
    # `weights_only=False` is explicit because a local resume checkpoint also
    # contains optimizer/scaler/argument metadata. Only load checkpoints from
    # trusted sources (the official VideoMAE release or your own output).
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        ckpt = torch.load(checkpoint_path, map_location='cpu')

    def tensor_state_candidates(value, path='checkpoint'):
        """Find state dicts in raw, model/state_dict, or nested encoder files."""
        if not isinstance(value, dict):
            return []
        direct = {key: item for key, item in value.items() if torch.is_tensor(item)}
        candidates = [(len(direct), path, direct)] if direct else []
        for key, item in value.items():
            if isinstance(item, dict):
                candidates.extend(tensor_state_candidates(item, f'{path}.{key}'))
        return candidates

    candidates = tensor_state_candidates(ckpt)
    if not candidates:
        raise ValueError(f'No tensor state_dict found in checkpoint: {checkpoint_path}')
    _, state_path, state = max(candidates, key=lambda item: item[0])
    # Official VideoMAE pretraining files commonly nest the encoder/backbone;
    # strip those container prefixes, then retain only shape-compatible tensors.
    def strip_prefix(key):
        while True:
            updated = re.sub(r'^(module\.|_orig_mod\.|model\.|backbone\.|encoder\.)', '', key)
            if updated == key:
                return key
            key = updated
    state = {strip_prefix(key): value for key, value in state.items()}
    target = model.state_dict()
    dropped = []
    for key in list(state):
        if key not in target or state[key].shape != target[key].shape:
            dropped.append((key, tuple(state[key].shape)))
            del state[key]
    loaded_keys = set(state)
    backbone_loaded = [key for key in loaded_keys if key.startswith(('patch_embed.', 'blocks.'))]
    if len(backbone_loaded) < 2:
        raise ValueError(f'{checkpoint_path} is incompatible: only {len(backbone_loaded)} VideoMAE backbone tensors match. '
                         'Refusing to train/evaluate with effectively random weights.')
    if resume and not {'head.weight', 'head.bias'} <= loaded_keys:
        raise ValueError(f'{checkpoint_path} is not a compatible DAiSEE resume checkpoint: its 4-class head is absent or mismatched.')
    msg = model.load_state_dict(state, strict=False)
    print(f'Loaded {checkpoint_path} (weights from {state_path}; {len(loaded_keys)} tensors matched); '
          f'dropped incompatible keys={dropped[:8]}; missing={list(msg.missing_keys)[:8]}')
    return ckpt


def reduce_sum(value, device):
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_initialized(): dist.all_reduce(tensor)
    return tensor.item()


def amp_context(device, enabled):
    """Use the non-deprecated AMP API when the installed PyTorch provides it."""
    dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        return torch.amp.autocast(device_type=device.type, enabled=enabled, dtype=dtype)
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)
    return torch.autocast(device_type='cpu', enabled=enabled, dtype=dtype)


def make_grad_scaler(device, enabled):
    enabled = enabled and device.type == 'cuda'
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        return torch.amp.GradScaler('cuda', enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def set_learning_rate(optimizer, args, epoch):
    """Epoch-indexed warmup + cosine schedule; exact after a checkpoint resume."""
    if epoch < args.warmup_epochs:
        factor = (epoch + 1) / max(args.warmup_epochs, 1)
    else:
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs - 1, 1)
        factor = 0.01 + 0.5 * (1. - 0.01) * (1. + np.cos(np.pi * min(progress, 1.)))
    for group in optimizer.param_groups:
        group['lr'] = args.lr * group.get('lr_scale', 1.0) * factor


def set_backbone_trainability(model, freeze_backbone):
    """Warm up only the new DAiSEE head before unfreezing pretrained features."""
    base_model = model.module if isinstance(model, DDP) else model
    for name, parameter in base_model.named_parameters():
        parameter.requires_grad = (not freeze_backbone or
                                   name.startswith(('head.', 'fc_norm.')))


def make_class_weights(records, args, device):
    """Conservative balancing for DAiSEE's extremely rare class 0.

    Raw inverse frequency would weight class 0 about 73x more than class 2,
    which caused degenerate all-class-0 predictions.  Capped square-root
    weighting retains minority signal without letting 34 clips dominate every
    update.  Class statistics are derived from training records only.
    """
    if not args.class_balanced:
        return None
    counts = np.asarray([sum(record['label'] == c for record in records)
                         for c in range(args.num_classes)], dtype=np.float64)
    if np.any(counts == 0):
        warnings.warn(f'Training classes with zero samples: {np.where(counts == 0)[0].tolist()}. '
                      'Their loss weights use the configured cap.')
    raw = np.sqrt(counts.max() / np.maximum(counts, 1.0))
    raw = np.minimum(raw, args.max_class_weight)
    weights = raw / raw.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def metrics(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred): cm[int(truth), int(pred)] += 1
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(n), where=cm.sum(0) != 0)
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(n), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(n), where=(precision + recall) != 0)
    total = cm.sum(); accuracy = np.trace(cm) / total if total else 0.
    expected = np.outer(cm.sum(1), cm.sum(0)) / total if total else cm.astype(float)
    weights = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / max((n - 1) ** 2, 1)
    qwk = 1 - (weights * cm).sum() / (weights * expected).sum() if (weights * expected).sum() else 0.
    return {'accuracy': accuracy, 'macro_f1': float(f1.mean()), 'mae': float(np.mean(np.abs(np.asarray(y_true)-np.asarray(y_pred)))),
            'qwk': float(qwk), 'precision': precision.tolist(), 'recall': recall.tolist(), 'f1': f1.tolist(), 'confusion_matrix': cm.tolist()}


def run_epoch(model, loader, criterion, optimizer, scaler, device, args, train):
    model.train(train); total_loss = total_correct = total_count = 0.; all_y, all_prob, all_ids = [], [], []
    if train: optimizer.zero_grad(set_to_none=True)
    for step, (x, y, _ids) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with amp_context(device, args.amp):
                logits = model(x); loss = criterion(logits, y)
            if train:
                group_start = (step // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(loader) - group_start)
                scaler.scale(loss / group_size).backward()
                if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                    scaler.unscale_(optimizer)
                    if args.clip_grad > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        total_loss += loss.detach().item() * len(y); total_correct += (logits.argmax(1) == y).sum().item(); total_count += len(y)
        all_y.extend(y.detach().cpu().tolist()); all_prob.extend(torch.softmax(logits.detach(), 1).cpu().tolist()); all_ids.extend(list(_ids))
    if dist.is_initialized():
        gathered = [None] * args.world_size; dist.all_gather_object(gathered, (all_y, all_prob, all_ids))
        all_y = [x for part in gathered for x in part[0]]; all_prob = [x for part in gathered for x in part[1]]; all_ids = [x for part in gathered for x in part[2]]
    # Average temporal test views by original ClipID before calculating metrics.
    # Validation has one view, so this is a no-op there.
    grouped = {}
    for target, probability, clip_id in zip(all_y, all_prob, all_ids):
        if clip_id in grouped and grouped[clip_id][0] != target:
            raise RuntimeError(f'Label leakage/corruption: ClipID {clip_id} has conflicting labels')
        if clip_id not in grouped: grouped[clip_id] = (target, [])
        grouped[clip_id][1].append(probability)
    all_y = [pair[0] for pair in grouped.values()]
    all_p = [int(np.asarray(pair[1]).mean(axis=0).argmax()) for pair in grouped.values()]
    denom = reduce_sum(total_count, device)
    loss = reduce_sum(total_loss, device) / max(denom, 1)
    report = metrics(all_y, all_p, args.num_classes)
    report.update(loss=loss)
    return report


def save_checkpoint(path, model, optimizer, scaler, epoch, args, best):
    torch.save({'model': (model.module if isinstance(model, DDP) else model).state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch,
                'best_macro_f1': best, 'args': vars(args)}, path)


def print_report(name, report):
    print(f'{name}: loss={report["loss"]:.4f}, accuracy={100*report["accuracy"]:.2f}%, macro-F1={100*report["macro_f1"]:.2f}%, MAE={report["mae"]:.4f}, QWK={report["qwk"]:.4f}')
    print('  precision=', [round(x, 4) for x in report['precision']], 'recall=', [round(x, 4) for x in report['recall']], 'F1=', [round(x, 4) for x in report['f1']])
    print('  confusion_matrix=', report['confusion_matrix'])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-path', default='/mnt/c/Users/puneet/Downloads/DAiSEE')
    p.add_argument('--output-dir', default='./outputs/daisee_videomae2')
    p.add_argument('--target-label', default='Engagement', help='One DAiSEE dimension, normally Engagement')
    p.add_argument('--subject-regex', default=r'^(\d{6})', help='Regex capture group identifying participant from ClipID')
    p.add_argument('--model', default='vit_base_patch16_224'); p.add_argument('--pretrained', default='')
    p.add_argument('--resume', default=''); p.add_argument('--eval-only', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Audit data and run one forward pass; do not train or save checkpoints')
    p.add_argument('--epochs', type=int, default=50); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4); p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--head-lr-multiplier', type=float, default=10.0,
                   help='Head LR / pretrained-backbone LR')
    p.add_argument('--head-warmup-epochs', type=int, default=2,
                   help='Train the new head only for this many initial epochs')
    p.add_argument('--weight-decay', type=float, default=0.05); p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--num-frames', type=int, default=16); p.add_argument('--sampling-rate', type=int, default=4)
    p.add_argument('--input-size', type=int, default=224); p.add_argument('--short-side-size', type=int, default=256)
    p.add_argument('--tubelet-size', type=int, default=2); p.add_argument('--num-classes', type=int, default=4)
    p.add_argument('--drop', type=float, default=0.); p.add_argument('--drop-path', type=float, default=0.1); p.add_argument('--head-drop', type=float, default=0.2); p.add_argument('--init-scale', type=float, default=0.001)
    p.add_argument('--aa', default='rand-m3-n2-mstd0.5-inc1'); p.add_argument('--train-interpolation', default='bicubic'); p.add_argument('--reprob', type=float, default=0.05)
    p.add_argument('--min-crop-scale', type=float, default=0.5,
                   help='Minimum retained image area for training crop')
    p.add_argument('--label-smoothing', type=float, default=0.0,
                   help='Keep at zero with class weighting unless deliberately tuning it')
    p.add_argument('--class-balanced', action='store_true', dest='class_balanced')
    p.add_argument('--no-class-balanced', action='store_false', dest='class_balanced')
    p.set_defaults(class_balanced=True)
    p.add_argument('--max-class-weight', type=float, default=5.0,
                   help='Cap before normalizing square-root class weights')
    p.add_argument('--test-clips', type=int, default=3); p.add_argument('--patience', type=int, default=12); p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--num-workers', type=int, default=4); p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='cuda')
    p.add_argument('--no-amp', action='store_false', dest='amp'); p.set_defaults(amp=True)
    p.add_argument('--with-checkpoint', action='store_true'); p.add_argument('--skip-video-verification', action='store_true')
    p.add_argument('--full-duplicate-audit', action='store_true', help='SHA-1 every video; slow but strongest duplicate check')
    return p.parse_args()


def main():
    args = parse_args(); setup_distributed(args)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu (slow) or run on the target GPU machine.')
    if args.device != 'cpu' and not args.device.startswith('cuda'):
        raise ValueError('--device currently supports cuda[:index] or cpu only')
    device = torch.device(f'cuda:{args.local_rank}' if args.distributed and args.device.startswith('cuda') else args.device)
    if args.test_clips < 1 or args.grad_accum < 1 or args.head_warmup_epochs < 0:
        raise ValueError('test-clips, grad-accum, and head-warmup-epochs must be valid')
    if not 0 < args.min_crop_scale <= 1 or args.head_lr_multiplier <= 0 or args.max_class_weight < 1:
        raise ValueError('min-crop-scale must be in (0,1], head-lr-multiplier > 0, and max-class-weight >= 1')
    seed_everything(args.seed, args.rank); root = Path(args.data_path).expanduser().resolve()
    if not root.is_dir(): raise FileNotFoundError(f'DAiSEE root does not exist on this machine: {root}')
    manifests, source = {}, {}
    for split in ('train', 'val', 'test'):
        manifests[split], source[split] = build_manifest(root, split, args.target_label, args)
    audit = audit_manifests(manifests, args)
    if is_main(args):
        print('Official DAiSEE sources:', json.dumps(source, indent=2)); print('Preflight:', json.dumps(audit, indent=2))
        for split, details in source.items():
            missing = details['missing_labelled_videos']
            if missing:
                warnings.warn(f'{split}: excluding {len(missing)} missing labelled video(s): {missing[:20]}. '
                              'Their labels are not used for training or metrics; see data_audit.json.')
        if not args.skip_video_verification: verify_videos([r for m in manifests.values() for r in m])
    if args.distributed: dist.barrier()
    if is_main(args):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / 'data_audit.json', 'w') as fh:
            json.dump({'source': source, 'audit': audit}, fh, indent=2)
    if args.distributed: dist.barrier()
    train_ds = DAiSEEDataset(manifests['train'], args, train=True)
    val_ds = DAiSEEDataset(manifests['val'], args, train=False)
    test_ds = DAiSEEDataset(manifests['test'], args, train=False, test_clips=args.test_clips)
    sampler = DistributedSampler(train_ds, shuffle=True) if args.distributed else None
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0, worker_init_fn=worker_init(args.seed))
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, drop_last=False, **common)
    # Distributed evaluation is intentionally avoided: repeating/padding samples biases macro-F1.
    val_loader, test_loader = DataLoader(val_ds, shuffle=False, **common), DataLoader(test_ds, shuffle=False, **common)
    model = make_model(args)
    if args.pretrained: load_weights(model, args.pretrained)
    elif not args.resume and is_main(args):
        warnings.warn('No --pretrained checkpoint supplied: training starts from random weights and is not the recommended DAiSEE protocol.')
    head_parameters, backbone_parameters = [], []
    for name, parameter in model.named_parameters():
        (head_parameters if name.startswith(('head.', 'fc_norm.')) else backbone_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': backbone_parameters, 'lr': args.lr, 'lr_scale': 1.0},
        {'params': head_parameters, 'lr': args.lr * args.head_lr_multiplier,
         'lr_scale': args.head_lr_multiplier},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scaler = make_grad_scaler(device, args.amp)
    start, best = 0, -float('inf')
    if args.resume:
        ckpt = load_weights(model, args.resume, resume=True)
        if not args.eval_only and 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer']); scaler.load_state_dict(ckpt.get('scaler', {})); start = ckpt.get('epoch', -1) + 1; best = ckpt.get('best_macro_f1', best)
    model.to(device)
    if args.distributed: model = DDP(model, device_ids=[args.local_rank] if device.type == 'cuda' else None)
    weight = make_class_weights(manifests['train'], args, device)
    if is_main(args):
        print('Class weights:', None if weight is None else [round(x, 4) for x in weight.detach().cpu().tolist()])
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    if args.dry_run:
        x, y, ids = next(iter(train_loader))
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        model.eval()
        with torch.no_grad():
            with amp_context(device, args.amp):
                logits = model(x)
        if tuple(logits.shape) != (len(y), args.num_classes):
            raise RuntimeError(f'Bad model output shape {tuple(logits.shape)}; expected ({len(y)}, {args.num_classes})')
        if is_main(args):
            print(f'DRY RUN PASSED: input={tuple(x.shape)}, labels={tuple(y.shape)}, '
                  f'logits={tuple(logits.shape)}, example_ids={list(ids[:3])}. No training was performed.')
        return
    if args.eval_only:
        if not (args.resume or args.pretrained):
            raise ValueError('--eval-only requires --resume (fine-tuned checkpoint) or --pretrained')
        report = run_epoch(model, test_loader, criterion, optimizer, scaler, device, args, False)
        if is_main(args): print_report('TEST', report); json.dump(report, open(Path(args.output_dir) / 'test_metrics.json', 'w'), indent=2)
        return
    stale = 0
    for epoch in range(start, args.epochs):
        if sampler: sampler.set_epoch(epoch)
        freeze_backbone = epoch < args.head_warmup_epochs
        set_backbone_trainability(model, freeze_backbone)
        if is_main(args):
            print(f'EPOCH {epoch + 1}: backbone={"frozen (head warm-up)" if freeze_backbone else "trainable"}')
        set_learning_rate(optimizer, args, epoch)
        train_report = run_epoch(model, train_loader, criterion, optimizer, scaler, device, args, True)
        val_report = run_epoch(model, val_loader, criterion, optimizer, scaler, device, args, False)
        if is_main(args):
            print_report(f'EPOCH {epoch + 1} TRAIN', train_report); print_report(f'EPOCH {epoch + 1} VAL', val_report)
            row = {'epoch': epoch + 1, 'lr': optimizer.param_groups[0]['lr'], 'train': train_report, 'val': val_report}
            with open(Path(args.output_dir) / 'metrics.jsonl', 'a') as fh: fh.write(json.dumps(row) + '\n')
            if val_report['macro_f1'] > best:
                best, stale = val_report['macro_f1'], 0; save_checkpoint(Path(args.output_dir) / 'best.pth', model, optimizer, scaler, epoch, args, best)
            else: stale += 1
            save_checkpoint(Path(args.output_dir) / 'last.pth', model, optimizer, scaler, epoch, args, best)
        if args.distributed:
            flag = torch.tensor([stale >= args.patience], device=device); dist.broadcast(flag, 0)
            if flag.item(): break
        elif stale >= args.patience: break
    if args.distributed: dist.barrier()
    load_weights(model.module if isinstance(model, DDP) else model, str(Path(args.output_dir) / 'best.pth'))
    report = run_epoch(model, test_loader, criterion, optimizer, scaler, device, args, False)
    if is_main(args):
        print_report('TEST (best validation macro-F1 checkpoint)', report)
        with open(Path(args.output_dir) / 'test_metrics.json', 'w') as fh: json.dump(report, fh, indent=2)


if __name__ == '__main__':
    main()
