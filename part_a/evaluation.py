"""Frozen-encoder evaluation: feature extraction, k-NN, and the linear probe.

k-NN is new in Part A (the assignment's first protocol). The linear probe trains lec5's
EncoderClassifier with the encoder frozen, on features computed once.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

import config
from augmentations import normalize, test_corruption, to_float
from model import encode


def device_of(module):
    for t in list(module.parameters()) + list(module.buffers()):
        return t.device
    return torch.device("cpu")


@torch.no_grad()
def extract_features(encoder, x_uint8, batch_size=1024, layer="h", corruption=None, severity=0, seed=0):
    """Run a frozen encoder over uint8 images [N, 3, 64, 64] -> float features [N, D] on the CPU.

    corruption and severity apply a test-time shift (augmentations.test_corruption) first.
    """
    was_training = encoder.training
    encoder.eval()
    dev = device_of(encoder)
    gen = torch.Generator().manual_seed(seed)
    feats = []
    for i in range(0, len(x_uint8), batch_size):
        x01 = to_float(x_uint8[i:i + batch_size].to(dev))
        if corruption is not None and severity > 0:
            x01 = test_corruption(x01, corruption, severity, generator=gen)
        feats.append(encode(encoder, normalize(x01), layer).float().cpu())
    encoder.train(was_training)
    return torch.cat(feats)


@torch.no_grad()
def knn_predict(bank_f, bank_y, query_f, num_classes, k=20, chunk=1024):
    """Similarity-weighted k-NN on the cosine similarity of l2-normalized features.

    Each of the k nearest bank items votes for its class with its cosine similarity.
    Returns class scores [Nq, C]: top-1 is the argmax, top-5 the 5 largest.
    """
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    bank = F.normalize(bank_f.to(dev), dim=1)
    bank_y = bank_y.to(dev)
    out = []
    for i in range(0, len(query_f), chunk):
        q = F.normalize(query_f[i:i + chunk].to(dev), dim=1)
        sim = q @ bank.T  # [n, N_bank]
        top_sim, top_idx = sim.topk(k, dim=1)  # [n, k]
        votes = torch.zeros(q.size(0), num_classes, device=dev)
        votes.scatter_add_(1, bank_y[top_idx], top_sim)
        out.append(votes.cpu())
    return torch.cat(out)


def topk_acc(scores, y, k):
    return (scores.topk(k, dim=1).indices == y.view(-1, 1)).any(dim=1).float().mean().item()


def knn_eval(encoder, bank_x, bank_y, query_x, query_y, num_classes, k=20, layer="h", **corr):
    bank_f = extract_features(encoder, bank_x, layer=layer)
    query_f = extract_features(encoder, query_x, layer=layer, **corr)
    scores = knn_predict(bank_f, bank_y, query_f, num_classes, k=k)
    return {"top1": topk_acc(scores, query_y, 1), "top5": topk_acc(scores, query_y, 5), "pred": scores.argmax(1)}


def train_linear_probe(train_f, train_y, val_f, val_y, num_classes, logger=None):
    """Train the head of a frozen EncoderClassifier for each weight decay in PROBE_WEIGHT_DECAYS,
    and keep the one with the best validation accuracy.

    Returns (best_model, best_weight_decay, {weight_decay: val_top1}).
    """
    from train import make_trainer  # late import: train.py imports this module
    from utils import build_model

    mean, std = train_f.mean(0), train_f.std(0) + 1e-6
    batch_size = min(config.PROBE_BATCH_SIZE, len(train_y))
    train_dl = DataLoader(TensorDataset(train_f, train_y), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(val_f, val_y), batch_size=4096)
    # 1% of the labels is only a few batches per epoch: train for at least PROBE_MIN_STEPS steps.
    epochs = max(config.PROBE_EPOCHS, math.ceil(config.PROBE_MIN_STEPS / len(train_dl)))
    best, best_wd, scores = None, None, {}
    for wd in config.PROBE_WEIGHT_DECAYS:
        pl.seed_everything(config.SEED, verbose=False)
        # The encoder is not needed (features are precomputed), so a random one keeps this light.
        probe = build_model("linear_probe", num_classes, epochs, encoder_path="random",
                            in_features=train_f.shape[1], feat_mean=mean, feat_std=std, probe_weight_decay=wd)
        trainer = make_trainer(epochs, logger=logger, precision="32", enable_progress_bar=False)
        trainer.fit(probe, train_dl, val_dl)
        acc = trainer.validate(probe, val_dl, verbose=False)[0]["val_accuracy"]
        scores[wd] = acc
        if best is None or acc > scores[best_wd]:
            best, best_wd = probe, wd
    return best, best_wd, scores


@torch.no_grad()
def probe_predict(probe, feats):
    probe.eval()
    return probe(feats.to(probe.head.weight.device)).float().cpu()


def per_class_accuracy(pred, y, num_classes):
    pred, y = np.asarray(pred), np.asarray(y)
    return np.array([(pred[y == c] == c).mean() for c in range(num_classes)])
