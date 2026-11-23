"""Tiny-ImageNet (pretraining and in-domain evaluation) and CIFAR-100 (transfer), 64x64 RGB.

lec5 had MnistDataModule, which read 28x28 images with torchvision and transformed each image
on the CPU. Here each dataset is converted once to a uint8 array [N, 64, 64, 3] and cached as
.npz in DATA_DIR. A whole split then sits in memory as a uint8 tensor (Tiny-ImageNet train is
about 1.2 GB), batches are sliced straight out of it, and float conversion, normalization,
corruption and augmentation run on the GPU on whole batches (augmentations.py). Colab has 2 CPU
cores, so per-image transforms in DataLoader workers would be the bottleneck.

Splits. lec5 used a seeded random 50k/10k split. The assignment asks for a class-balanced
hold-out, so every model sees exactly the same images:
  train : official train minus the hold-out (Tiny-ImageNet: 450/class = 90k)
  val   : 50 images per class held out of the official train split (10k), for all tuning
  test  : the official labelled validation split (Tiny-ImageNet "valid", 10k), final numbers only
Label fractions (1%, 10%, 100%) are class-balanced subsets of train. lec5 kept the first
N_TARGET_TRAIN images instead, which is not balanced across classes.
"""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset

DATASETS = {
    # name: (num_classes, hold-out images per class for validation)
    "tiny_imagenet": (200, 50),
    "cifar100": (100, 50),
}

# Tiny-ImageNet statistics. The transfer set is normalized with the SAME statistics,
# because the frozen encoder only ever saw inputs on this scale.
MEAN = (0.4802, 0.4481, 0.3975)
STD = (0.2770, 0.2691, 0.2821)


# --------------------------------------------------------------------------------------
# One-time conversion to .npz
# --------------------------------------------------------------------------------------
def _cache_path(data_dir, name):
    return os.path.join(data_dir, f"{name}_64.npz")


STANFORD_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _prepare_tiny_imagenet_stanford(path, data_dir):
    """Same dataset from its original host (for networks that block huggingface.co).

    Labels follow sorted WordNet ids, as torchvision's ImageFolder would. The official 'val'
    folder (10k, labels in val_annotations.txt) is the test set, exactly like HF 'valid'.
    """
    import urllib.request
    import zipfile
    from PIL import Image

    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {STANFORD_URL} (~248 MB)")
        urllib.request.urlretrieve(STANFORD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        wnids = sorted(zf.read("tiny-imagenet-200/wnids.txt").decode().split())
        label = {w: i for i, w in enumerate(wnids)}

        def read(name):
            with zf.open(name) as f:
                return np.asarray(Image.open(f).convert("RGB"))

        train = sorted(n for n in names if n.startswith("tiny-imagenet-200/train/") and n.endswith(".JPEG"))
        train_x = np.stack([read(n) for n in train])
        train_y = np.array([label[n.split("/")[2]] for n in train], dtype=np.int64)

        ann = zf.read("tiny-imagenet-200/val/val_annotations.txt").decode().strip().splitlines()
        ann = sorted(line.split("\t")[:2] for line in ann)
        test_x = np.stack([read(f"tiny-imagenet-200/val/images/{f}") for f, _ in ann])
        test_y = np.array([label[w] for _, w in ann], dtype=np.int64)
    np.savez(path, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, class_names=np.array(wnids))


def _prepare_tiny_imagenet(path):
    from datasets import load_dataset  # Hugging Face; only needed once

    ds = load_dataset("zh-plus/tiny-imagenet")

    def to_arrays(split):
        n = len(split)
        x = np.empty((n, 64, 64, 3), dtype=np.uint8)
        y = np.empty(n, dtype=np.int64)
        for i, item in enumerate(split):
            x[i] = np.asarray(item["image"].convert("RGB"))  # a few images are grayscale
            y[i] = item["label"]
        return x, y

    train_x, train_y = to_arrays(ds["train"])
    test_x, test_y = to_arrays(ds["valid"])
    names = np.array(ds["train"].features["label"].names)  # WordNet ids, e.g. n01443537
    np.savez(path, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, class_names=names)


def _prepare_cifar100(path, data_dir):
    import torchvision

    def to_arrays(train):
        ds = torchvision.datasets.CIFAR100(os.path.join(data_dir, "cifar100_raw"), train=train, download=True)
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float()  # [N, 3, 32, 32]
        out = []
        for chunk in x.split(5000):  # 32x32 -> 64x64, as the assignment asks
            up = F.interpolate(chunk, size=64, mode="bilinear", align_corners=False, antialias=True)
            out.append(up.round().clamp(0, 255).to(torch.uint8))
        x = torch.cat(out).permute(0, 2, 3, 1).numpy()
        return x, np.asarray(ds.targets, dtype=np.int64), ds.classes

    train_x, train_y, names = to_arrays(True)
    test_x, test_y, _ = to_arrays(False)
    np.savez(path, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, class_names=np.array(names))


def prepare_dataset(name, data_dir, source="hf"):
    """source (Tiny-ImageNet only): 'hf' = zh-plus/tiny-imagenet (assignment default),
    'stanford' = the original zip. Both hold the same 100k train / 10k val images. Use ONE
    source for a whole project: the seeded hold-out split depends on the image order."""
    os.makedirs(data_dir, exist_ok=True)
    path = _cache_path(data_dir, name)
    if os.path.exists(path):
        return path
    print(f"Preparing {name} -> {path} (one time)")
    if name == "tiny_imagenet" and source == "stanford":
        _prepare_tiny_imagenet_stanford(path, data_dir)
    elif name == "tiny_imagenet":
        _prepare_tiny_imagenet(path)
    elif name == "cifar100":
        _prepare_cifar100(path, data_dir)
    else:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {list(DATASETS)}")
    return path


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
def holdout_split(labels, per_class, seed):
    """Class-balanced hold-out: `per_class` images of every class go to val."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        val_idx.append(idx[:per_class])
        train_idx.append(idx[per_class:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def balanced_subset(indices, labels, fraction=None, per_class=None, seed=0):
    """Class-balanced subset of `indices`.

    fraction=0.01 keeps ceil(1% of each class) images (Tiny-ImageNet: 5/class = 1,000 images).
    The same seed gives the same subset for every model, so all models see identical labels.
    """
    if fraction is None and per_class is None:
        return indices
    if fraction is not None and fraction >= 1.0 and per_class is None:
        return indices
    rng = np.random.default_rng(seed + 1)  # different stream from the hold-out split
    keep = []
    for c in np.unique(labels[indices]):
        idx = indices[labels[indices] == c]
        n = per_class if per_class is not None else max(1, math.ceil(fraction * len(idx)))
        keep.append(rng.permutation(idx)[:n])
    return np.sort(np.concatenate(keep))


# --------------------------------------------------------------------------------------
# DataModule
# --------------------------------------------------------------------------------------
def _batched_loader(x, y, batch_size, shuffle, drop_last, seed):
    """DataLoader that slices whole batches out of in-memory tensors (no per-item collate)."""
    ds = TensorDataset(x, y)
    if shuffle:
        sampler = RandomSampler(ds, generator=torch.Generator().manual_seed(seed))
    else:
        sampler = SequentialSampler(ds)
    batches = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    # batch_size=None: the dataset receives a list of indices and returns a whole batch.
    return DataLoader(ds, sampler=batches, batch_size=None, num_workers=0)


class TinyImageNetDataModule(pl.LightningDataModule):
    """Tiny-ImageNet or CIFAR-100 (64x64), batches of uint8 [B, 3, 64, 64] + int labels.

    label_fraction / train_per_class restrict the TRAIN split (class-balanced, seeded).
    Validation and test are never subsampled.
    """

    def __init__(self, dataset="tiny_imagenet", data_dir="dataset/", batch_size=256,
                 label_fraction=None, train_per_class=None, seed=42, drop_last=False, source="stanford"):
        super().__init__()
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset!r}. Choose from {list(DATASETS)}")
        self.dataset = dataset
        self.data_dir = data_dir
        self.source = source
        self.batch_size = batch_size
        self.label_fraction = label_fraction
        self.train_per_class = train_per_class
        self.seed = seed
        self.drop_last = drop_last
        self.num_classes, self.holdout_per_class = DATASETS[dataset]
        self._loaded = False

    def prepare_data(self):
        prepare_dataset(self.dataset, self.data_dir, self.source)

    def setup(self, stage=None):
        if self._loaded:
            return
        data = np.load(_cache_path(self.data_dir, self.dataset))
        x_all = torch.from_numpy(data["train_x"]).permute(0, 3, 1, 2).contiguous()  # [N, 3, 64, 64] uint8
        y_all = data["train_y"]
        train_idx, val_idx = holdout_split(y_all, self.holdout_per_class, self.seed)
        self.full_train_idx = train_idx
        train_idx = balanced_subset(train_idx, y_all, self.label_fraction, self.train_per_class, self.seed)

        y_t = torch.from_numpy(y_all)
        self.train_x, self.train_y = x_all[train_idx], y_t[train_idx]
        self.val_x, self.val_y = x_all[val_idx], y_t[val_idx]
        # Full (un-subsampled) train split: the k-NN memory bank always uses all of it.
        if len(train_idx) == len(self.full_train_idx):  # no subset: share memory (~1 GB) instead of copying
            self.bank_x, self.bank_y = self.train_x, self.train_y
        else:
            self.bank_x, self.bank_y = x_all[self.full_train_idx], y_t[self.full_train_idx]
        self.test_x = torch.from_numpy(data["test_x"]).permute(0, 3, 1, 2).contiguous()
        self.test_y = torch.from_numpy(data["test_y"])
        names = data["class_names"] if "class_names" in data.files else np.arange(self.num_classes).astype(str)
        self.class_names = [str(n) for n in names]
        self._loaded = True
        print(f"[{self.dataset}] train {len(self.train_y)} | val {len(self.val_y)} | "
              f"test {len(self.test_y)} | bank {len(self.bank_y)}")

    def train_dataloader(self):
        return _batched_loader(self.train_x, self.train_y, self.batch_size, True, self.drop_last, self.seed)

    def val_dataloader(self):
        return _batched_loader(self.val_x, self.val_y, 1024, False, False, self.seed)

    def test_dataloader(self):
        return _batched_loader(self.test_x, self.test_y, 1024, False, False, self.seed)

    def split(self, name):
        """(x_uint8, y) of one split: 'train', 'val', 'test' or 'bank'."""
        return getattr(self, f"{name}_x"), getattr(self, f"{name}_y")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download and cache a dataset as 64x64 uint8 .npz")
    p.add_argument("dataset", choices=list(DATASETS))
    p.add_argument("--data_dir", default="dataset/")
    p.add_argument("--source", default="hf", choices=["hf", "stanford"])
    a = p.parse_args()
    print(prepare_dataset(a.dataset, a.data_dir, a.source))
