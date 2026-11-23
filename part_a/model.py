import copy
import math

import torch
import torch.nn.functional as F
from torch import nn, optim
import pytorch_lightning as pl
import torchmetrics
import torchvision

from augmentations import augment_finetune, corrupt, normalize, to_float
from losses import nt_xent_loss, triplet_loss


def build_encoder(latent_dim):
    """The small encoder of lec3-lec5, for 64x64 RGB instead of 28x28 grayscale.

    The layers are the same: two stride-2 convolutions, then a linear layer that compresses
    the last map into h. Only the sizes change: 3 input channels, and 64 -> 32 -> 16, so the
    linear layer reads 64*16*16 numbers. lec5 wrote these lines out in every model class;
    here they live in one function, so every model is guaranteed the same layers in the same
    order, and an encoder file saved by one model loads into any other.
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [B, 3, 64, 64] -> [B, 32, 32, 32]
        nn.ReLU(),  # [B, 32, 32, 32] -> [B, 32, 32, 32]
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 32, 32, 32] -> [B, 64, 16, 16]
        nn.ReLU(),  # [B, 64, 16, 16] -> [B, 64, 16, 16]
        nn.Flatten(),  # [B, 64, 16, 16] -> [B, 16384]
        nn.Linear(64 * 16 * 16, latent_dim),  # [B, 16384] -> [B, latent_dim], this output is h
    )


class PixelEncoder(nn.Module):
    """No learning at all: h is the normalized image average-pooled to 16x16 (768 numbers).

    A sanity baseline for k-NN and the linear probe: is a learned h better than raw pixels?
    """

    def __init__(self, size=16):
        super().__init__()
        self.size = size
        self.register_buffer("_device_anchor", torch.zeros(1))  # so .to(device) has something to move

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.size).flatten(1)  # [B, 3, 64, 64] -> [B, 768]


def load_encoder(encoder_path, latent_dim):
    """encoder_path: a file saved by a pretrain run, "random" (or None) for random init, or "pixels".

    The width of h is read from the file, so the protocol needs no LATENT_DIM of its own.
    """
    if encoder_path == "pixels":
        return PixelEncoder()
    if encoder_path in (None, "", "random"):
        return build_encoder(latent_dim)  # random init, seeded by pl.seed_everything
    state_dict = torch.load(encoder_path, map_location="cpu", weights_only=True)
    encoder = build_encoder(state_dict["5.weight"].shape[0])
    encoder.load_state_dict(state_dict)
    return encoder


def encode(encoder, x_normalized, layer="h"):
    """Features of a batch. layer="h": the bottleneck output. layer="conv": the last conv map
    (after its ReLU) average-pooled to 4x4, 64*4*4 = 1024 numbers. The second one answers
    the assignment's question of which layer to use as h."""
    if layer == "h" or isinstance(encoder, PixelEncoder):
        return encoder(x_normalized)
    if layer == "conv":
        conv_map = encoder[:4](x_normalized)  # Conv, ReLU, Conv, ReLU: [B, 64, 16, 16]
        return F.adaptive_avg_pool2d(conv_map, 4).flatten(1)  # [B, 1024]
    raise ValueError(f"Unknown layer {layer!r}. Use 'h' or 'conv'.")


def _cosine_schedule(optimizer, total_steps, warmup_steps=0):
    """Linear warm-up, then the learning rate follows half a cosine down to 0."""

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _adamw_cosine(module, params, learning_rate, weight_decay, warmup_epochs=0):
    """AdamW with a cosine schedule, stepped every batch. lec5 used Adam with a fixed lr.
    We tune weight decay, which needs AdamW, and the cosine decay makes the result less
    sensitive to the exact learning rate."""
    opt = optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    total = module.trainer.estimated_stepping_batches
    warmup = int(total * warmup_epochs / max(1, module.hparams.max_epochs))
    sched = _cosine_schedule(opt, total, warmup)
    return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _log_image_grid(module, images, tag, nrow):
    if module.logger is None or not hasattr(module.logger.experiment, "add_image"):
        return
    grid = torchvision.utils.make_grid(images.detach().float().clamp(0, 1).cpu(), nrow=nrow)
    module.logger.experiment.add_image(tag, grid, module.global_step)


class AutoEncoder(pl.LightningModule):
    """Denoising autoencoder (Vincent et al., 2008), grown from lec5's AutoEncoder.

    The encoder is build_encoder(). The decoder is its mirror image, as in lec5: a linear
    layer expands h back to 64x16x16, and two transposed convolutions invert the two stride-2
    convolutions (16 -> 32 -> 64). output_padding=1 lands the sizes on 32 and 64.

    What is new: the encoder sees a corrupted image x~ ~ q(x~|x), and the decoder must rebuild
    the clean x. With corruption="none" this is lec5's plain autoencoder again, which is the
    plain counterpart the assignment asks for.

    The loss is the mean squared error to the clean image in [0, 1] pixel space. lec5 compared
    normalized images. With [0, 1] pixels the error means the same thing for every corruption
    and every dataset, which the reconstruction-vs-k-NN plot needs.

    Logged on validation: val_loss is the training objective, || x - g(f(x~)) ||^2.
    val_recon_mse is || x - g(f(x)) ||^2 on clean inputs, so runs with different corruptions
    are compared on the same task. callbacks.KNNMonitor adds val_knn_top1, which selects the epoch.
    """

    def __init__(self, learning_rate, latent_dim, corruption="none", noise_level=0.0, mask_patch=8,
                 weight_decay=0.0, max_epochs=40):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate

        self.encoder = build_encoder(latent_dim)  # [B, 3, 64, 64] -> [B, latent_dim]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 16 * 16),  # [B, latent_dim] -> [B, 16384]
            nn.ReLU(),  # [B, 16384] -> [B, 16384]
            nn.Unflatten(1, (64, 16, 16)),  # [B, 16384] -> [B, 64, 16, 16]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 32, 32, 32]
            nn.ReLU(),  # [B, 32, 32, 32] -> [B, 32, 32, 32]
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [B, 3, 64, 64]
        )  # no activation at the end, as in lec5: the output is compared with [0, 1] pixels directly

    def forward(self, x01):
        z = self.encoder(normalize(x01))  # [B, 3, 64, 64] -> [B, latent_dim]
        return self.decoder(z)  # [B, latent_dim] -> [B, 3, 64, 64]

    def _corrupt(self, x01):
        return corrupt(x01, self.hparams.corruption, self.hparams.noise_level, self.hparams.mask_patch)

    def training_step(self, batch, batch_idx):
        x01 = to_float(batch[0])  # the label is unused
        x_tilde = self._corrupt(x01)
        loss = F.mse_loss(self(x_tilde), x01)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x01 = to_float(batch[0])
        x_tilde = self._corrupt(x01)
        recon = self(x_tilde)
        self.log("val_loss", F.mse_loss(recon, x01), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_mse", F.mse_loss(self(x01), x01), on_step=False, on_epoch=True)
        if batch_idx == 0 and not self.trainer.sanity_checking:
            self._log_reconstructions(x01, x_tilde, recon, "val_reconstructions")
        return recon

    def test_step(self, batch, batch_idx):
        x01 = to_float(batch[0])
        self.log("test_loss", F.mse_loss(self(self._corrupt(x01)), x01), on_step=False, on_epoch=True)
        self.log("test_recon_mse", F.mse_loss(self(x01), x01), on_step=False, on_epoch=True)

    def _log_reconstructions(self, x, x_tilde, recon, tag):
        """Rows: clean input, corrupted input, reconstruction."""
        n = min(8, x.size(0))
        _log_image_grid(self, torch.cat([x[:n], x_tilde[:n], recon[:n]]), tag, nrow=n)

    def configure_optimizers(self):
        return _adamw_cosine(self, self.parameters(), self.lr, self.hparams.weight_decay)


class EncoderClassifier(pl.LightningModule):
    """Pretrained encoder plus a new linear head. Used for linear probe and finetune, as in lec5.

    freeze_encoder=True trains only the head (linear probe). False trains everything (finetune).

    Two additions for Tiny-ImageNet:
    * The linear probe runs on features the frozen encoder computed once
      (evaluation.train_linear_probe). A frozen encoder gives the same h every epoch, so
      recomputing it would only cost time. A batch can therefore be images [B, 3, 64, 64] or
      features [B, in_features]. Features are standardized with the train-set mean and std,
      a fixed affine map, so the classifier stays linear.
    * Fine-tuning uses random crop + flip (augmentations.augment_finetune), AdamW with a short
      warm-up and a cosine schedule, and reports top-1 and top-5 accuracy.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, encoder_path, freeze_encoder=False,
                 weight_decay=0.0, max_epochs=20, warmup_epochs=1, augment=True,
                 in_features=None, feat_mean=None, feat_std=None):
        super().__init__()
        self.save_hyperparameters(ignore=["feat_mean", "feat_std"])
        self.lr = learning_rate
        self.freeze_encoder = freeze_encoder

        # Same layers as AutoEncoder.encoder, with the weights of encoder_path.
        self.encoder = load_encoder(encoder_path, latent_dim)
        in_features = in_features or latent_dim
        self.head = nn.Linear(in_features, num_classes)  # [B, in_features] -> [B, num_classes]
        self.register_buffer("feat_mean", feat_mean if feat_mean is not None else torch.zeros(in_features))
        self.register_buffer("feat_std", feat_std if feat_std is not None else torch.ones(in_features))

        # eval() before freezing: a frozen encoder must not update BatchNorm stats
        # or dropout. train() below keeps that, because Lightning calls train()
        # at the start of every epoch and would otherwise undo it.
        if self.freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.loss_fn = nn.CrossEntropyLoss()
        acc = lambda k: torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=k)
        self.train_accuracy = acc(1)
        self.val_accuracy, self.val_top5 = acc(1), acc(5)
        self.test_accuracy, self.test_top5 = acc(1), acc(5)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x):
        if x.dim() == 2:  # precomputed features of the frozen encoder
            return self.head((x - self.feat_mean) / self.feat_std)
        z = self.encoder(normalize(x))  # [B, 3, 64, 64] -> [B, latent_dim]
        return self.head(z)  # [B, latent_dim] -> [B, num_classes]

    def _inputs(self, x, train):
        if x.dim() == 2:
            return x
        x01 = to_float(x)
        if train and self.hparams.augment:
            x01 = augment_finetune(x01)
        return x01

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self._inputs(x, train=True)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        self.train_accuracy(scores, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True)
        if batch_idx % 100 == 0 and x.dim() == 4:
            _log_image_grid(self, x[:8], "train_images", nrow=8)
        return loss

    def _eval_step(self, batch, stage):
        x, y = batch
        scores = self.forward(self._inputs(x, train=False))
        top1, top5 = getattr(self, f"{stage}_accuracy"), getattr(self, f"{stage}_top5")
        top1(scores, y)
        top5(scores, y)
        self.log(f"{stage}_loss", self.loss_fn(scores, y), on_step=False, on_epoch=True)
        self.log(f"{stage}_accuracy", top1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_top5", top5, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return torch.argmax(self.forward(self._inputs(x, train=False)), dim=1)

    def configure_optimizers(self):
        # Linear probing: only the head is trainable, so only the head is optimized.
        params = [p for p in self.parameters() if p.requires_grad]
        return _adamw_cosine(self, params, self.lr, self.hparams.weight_decay, self.hparams.warmup_epochs)


class SupervisedEncoder(EncoderClassifier):
    """Small encoder plus a linear head, both trained from scratch with labels.

    Same encoder layers as AutoEncoder, with cross-entropy instead of a decoder. It is an
    EncoderClassifier with no pretrained file, so it trains with exactly the recipe that
    fine-tuning uses. That makes it the fair "supervised from scratch" baseline: on the same
    label subsets as finetune, and on all labels as the supervised reference encoder.
    """

    def __init__(self, learning_rate, num_classes, latent_dim, weight_decay=0.0, max_epochs=20,
                 warmup_epochs=1, augment=True):
        super().__init__(learning_rate, num_classes, latent_dim, encoder_path=None, freeze_encoder=False,
                         weight_decay=weight_decay, max_epochs=max_epochs, warmup_epochs=warmup_epochs,
                         augment=augment)


# --------------------------------------------------------------------------------------
# SimCLR (lec4) and MoCo (lec5), unchanged apart from the shared 64x64 RGB encoder.
# Part A does not use them. Part B brings back their modes and the two-view data path.
# --------------------------------------------------------------------------------------
class SimCLR(pl.LightningModule):
    def __init__(self, learning_rate, latent_dim, projector_dim, temperature, margin, loss_name):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.temperature = temperature
        self.margin = margin
        self.loss_name = loss_name

        # Same layers, same order as AutoEncoder.encoder.
        self.encoder = build_encoder(latent_dim)  # [B, 3, 64, 64] -> [B, latent_dim]
        # g(h): Linear -> ReLU -> Linear, with a 4x wider hidden layer.
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 4 * projector_dim),  # [B, latent_dim] -> [B, 4 * projector_dim]
            nn.ReLU(inplace=True),  # [B, 4 * projector_dim] -> [B, 4 * projector_dim]
            nn.Linear(4 * projector_dim, projector_dim),  # [B, 4 * projector_dim] -> [B, projector_dim]
        )

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, "train")
        if batch_idx % 100 == 0:
            (xi, xj), _ = batch
            self._log_views(xi, xj)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def _common_step(self, batch, mode):
        (xi, xj), _ = batch  # two augmented views, each [B, 3, 64, 64]; the label is unused

        # representations
        h1 = self.encoder(xi)  # [B, 3, 64, 64] -> [B, latent_dim]
        h2 = self.encoder(xj)  # [B, 3, 64, 64] -> [B, latent_dim]

        z1 = self.projector(h1)  # [B, latent_dim] -> [B, projector_dim]
        z2 = self.projector(h2)  # [B, latent_dim] -> [B, projector_dim]

        if self.loss_name == "nt_xent":
            loss = nt_xent_loss(z1, z2, self.temperature)
        elif self.loss_name == "triplet":
            loss = triplet_loss(z1, z2, self.margin)
        else:
            raise ValueError(f"Unknown loss {self.loss_name!r}. Use 'nt_xent' or 'triplet'.")

        self.log(
            f"{mode}_loss",
            loss,
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        # How close a positive pair is, compared with the other images in the batch.
        stats = self._pair_stats(z1, z2)
        self.log(
            f"{mode}_pos_sim",
            stats.pop("pos_sim"),
            on_step=(mode == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        self.log_dict(
            {f"{mode}_{name}": value for name, value in stats.items()},
            on_step=(mode == "train"),
            on_epoch=True,
        )
        return loss

    def _pair_stats(self, z1, z2):
        """Scalars for TensorBoard. They are not part of the loss."""
        z1 = F.normalize(z1.detach(), dim=-1)
        z2 = F.normalize(z2.detach(), dim=-1)
        # Cosine similarity of every anchor in z1 with every view in z2.
        pair = torch.mm(z1, z2.t())  # [B, B]
        eye = torch.eye(pair.shape[0], dtype=torch.bool, device=pair.device)
        n_negatives = pair.shape[0] * (pair.shape[0] - 1)
        stats = {
            "pos_sim": pair.diag().mean(),
            "neg_sim": pair.masked_fill(eye, 0).sum() / n_negatives,
        }

        if self.loss_name == "nt_xent":
            # Rank of the matching view among the other 2B - 1 rows. 0 means it is the nearest.
            out = torch.cat([z1, z2], dim=0)
            sim = torch.mm(out, out.t())
            n = sim.shape[0]
            self_mask = torch.eye(n, dtype=torch.bool, device=sim.device)
            index = torch.arange(n, device=sim.device)
            positive = sim[index, (index + n // 2) % n]
            rank = ((sim > positive.unsqueeze(1)) & ~self_mask).sum(dim=1)
            stats["acc_top1"] = (rank == 0).float().mean()
            stats["acc_top5"] = (rank < 5).float().mean()
            stats["acc_mean_pos"] = (rank + 1).float().mean()
        elif self.loss_name == "triplet":
            # Share of negatives that still violate the margin.
            dist = (2 - 2 * pair).clamp(min=0)
            positive = dist.diag().unsqueeze(1)
            violate = (positive - dist + self.margin > 0) & ~eye
            stats["active"] = violate.float().sum() / (~eye).float().sum()
        return stats

    def _log_views(self, xi, xj):
        """Top row: xi. Bottom row: xj."""
        if self.logger is None:
            return
        n = min(8, xi.size(0))
        images = torch.cat([xi[:n], xj[:n]]).detach().float().cpu()
        grid = torchvision.utils.make_grid(images, nrow=n)
        self.logger.experiment.add_image("contrastive_views", grid, self.global_step)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class MoCo(pl.LightningModule):
    def __init__(self, learning_rate, latent_dim, projector_dim, K, m, T):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.K = K
        self.m = m
        self.T = T

        # Same layers, same order as AutoEncoder.encoder and SimCLR.encoder.
        self.encoder = build_encoder(latent_dim)  # [B, 3, 64, 64] -> [B, latent_dim]
        # MoCo v2 head (the repo's mlp=True). v1 is a single linear layer.
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 4 * projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * projector_dim, projector_dim),
        )
        self.key_encoder = copy.deepcopy(self.encoder)
        self.key_projector = copy.deepcopy(self.projector)
        for p in list(self.key_encoder.parameters()) + list(self.key_projector.parameters()):
            p.requires_grad = False

        # Queue of K negatives. Each column is one key, shape [projector_dim, K].
        self.register_buffer("queue", F.normalize(torch.randn(projector_dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """key = m * key + (1 - m) * query. Same update as the repo."""
        query = list(self.encoder.parameters()) + list(self.projector.parameters())
        key = list(self.key_encoder.parameters()) + list(self.key_projector.parameters())
        for param_q, param_k in zip(query, key):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Replace K columns at the pointer with this batch of keys, then advance."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # pointer always lands on a slot boundary
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def _common_step(self, batch, mode):
        """One MoCo step. Logit 0 is the positive key. The other K logits are the queue."""
        (im_q, im_k), _ = batch

        # NOTE: for student: why do we normelize the output of the projector? 
        q = F.normalize(self.projector(self.encoder(im_q)), dim=1)  # [B, C], with gradient

        with torch.no_grad():  # the key is not trained
            if self.training:
                self._momentum_update_key_encoder()
            k = F.normalize(self.key_projector(self.key_encoder(im_k)), dim=1)  # [B, C]

        l_pos = (q * k).sum(dim=1, keepdim=True)  # [B, 1], the matching key
        l_neg = q @ self.queue.clone().detach()  # [B, K], the queue
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T  # [B, 1+K]

        if self.training:  # val and test read the queue, they do not write it
            self._dequeue_and_enqueue(k)

        # Cross-entropy for class 0, the positive key:
        # -log( exp(logit_0) / sum_j exp(logit_j) ) = -logit_0 + logsumexp(logits)
        loss = -(logits[:, 0] - torch.logsumexp(logits, dim=1)).mean()
        self.log(f"{mode}_loss", loss, on_step=(mode == "train"), on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.projector.parameters())
        return optim.Adam(params, lr=self.lr)
