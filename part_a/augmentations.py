"""Augmentations and corruptions, applied on the GPU (or MPS / CPU) to whole batches.

lec5 built torchvision transforms that ran on one PIL image at a time. Here the DataModule
hands out uint8 batches and everything below works on a whole batch at once. The SimCLR
two-view augmentation of lec4/lec5 returns in Part B, in the same batched form.

Images arrive from the DataModule as uint8 [B, 3, 64, 64]. Everything here works on float
images in [0, 1]; `normalize` is applied last, right before the encoder.

Three groups:
  * corrupt()           : q(x~ | x) for the denoising autoencoder (training-time corruption)
  * augment_finetune()  : light augmentation for supervised fine-tuning (crop + flip)
  * test_corruption()   : ImageNet-C style test-time shifts for the robustness test (A4 bonus)
"""
import torch
import torch.nn.functional as F

from datamodule import MEAN, STD


def to_float(x_uint8):
    return x_uint8.float().div_(255.0)


def normalize(x01):
    mean = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def denormalize(xn):
    mean = torch.tensor(MEAN, device=xn.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=xn.device).view(1, 3, 1, 1)
    return (xn * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------------------
# DAE corruption q(x~ | x)
# --------------------------------------------------------------------------------------
CORRUPTIONS = ("none", "gaussian", "mask", "salt_pepper")


def corrupt(x01, kind, level, patch=8):
    """Corrupt a batch of [0, 1] images. `level` means:

    gaussian    : std of additive N(0, level^2) pixel noise (Vincent et al., 2008/2010)
    mask        : fraction of patch x patch blocks set to 0 (block masking; a pixel-level mask
                  at 64x64 is trivially in-painted from neighbours, blocks force context use)
    salt_pepper : fraction of pixels set to 0 or 1 with equal probability
    none        : identity (vanilla autoencoder, the plain counterpart)
    """
    if kind == "none" or level <= 0:
        return x01
    if kind == "gaussian":
        return (x01 + level * torch.randn_like(x01)).clamp(0, 1)
    if kind == "mask":
        b, _, h, w = x01.shape
        keep = (torch.rand(b, 1, h // patch, w // patch, device=x01.device) >= level).float()
        keep = F.interpolate(keep, scale_factor=patch, mode="nearest")
        return x01 * keep
    if kind == "salt_pepper":
        b, _, h, w = x01.shape
        u = torch.rand(b, 1, h, w, device=x01.device)  # same pixel location in all channels
        salt = (u < level / 2).float()
        pepper = ((u >= level / 2) & (u < level)).float()
        return x01 * (1 - salt - pepper) + salt
    raise ValueError(f"Unknown corruption {kind!r}. Choose from {CORRUPTIONS}")


# --------------------------------------------------------------------------------------
# Fine-tuning augmentation
# --------------------------------------------------------------------------------------
def augment_finetune(x01, pad=4):
    """Random crop with reflect padding + horizontal flip, independently per image."""
    b, _, h, w = x01.shape
    padded = F.pad(x01, (pad, pad, pad, pad), mode="reflect")
    dx = torch.randint(0, 2 * pad + 1, (b,), device=x01.device)
    dy = torch.randint(0, 2 * pad + 1, (b,), device=x01.device)
    rows = (dy.view(b, 1) + torch.arange(h, device=x01.device).view(1, h))  # [B, H]
    cols = (dx.view(b, 1) + torch.arange(w, device=x01.device).view(1, w))  # [B, W]
    bidx = torch.arange(b, device=x01.device).view(b, 1, 1)
    out = padded.permute(0, 2, 3, 1)[bidx, rows.view(b, h, 1), cols.view(b, 1, w)]  # [B, H, W, C]
    out = out.permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x01.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out.contiguous()


# --------------------------------------------------------------------------------------
# Test-time corruptions (robustness bonus)
# --------------------------------------------------------------------------------------
TEST_CORRUPTIONS = ("gaussian_noise", "salt_pepper", "occlusion", "blur", "contrast", "brightness")
# severity 1..5, in the spirit of ImageNet-C (Hendrycks & Dietterich, 2019)
_SEVERITY = {
    "gaussian_noise": [0.04, 0.08, 0.12, 0.18, 0.26],  # noise std
    "salt_pepper": [0.01, 0.03, 0.06, 0.10, 0.17],     # fraction of pixels
    "occlusion": [0.1, 0.2, 0.3, 0.4, 0.5],            # fraction of 8x8 blocks dropped
    "blur": [0.5, 0.8, 1.1, 1.5, 2.0],                 # gaussian sigma (pixels)
    "contrast": [0.75, 0.6, 0.45, 0.3, 0.2],           # contrast factor
    "brightness": [0.1, 0.2, 0.3, 0.4, 0.5],           # additive brightness
}


def _gaussian_blur(x01, sigma):
    radius = max(1, int(round(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x01.device, dtype=x01.dtype)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    c = x01.shape[1]
    x = F.pad(x01, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return x


def test_corruption(x01, kind, severity, generator=None):
    """Apply one test-time corruption at severity 1..5 (0 = clean)."""
    if severity == 0:
        return x01
    s = _SEVERITY[kind][severity - 1]
    if kind == "gaussian_noise":
        noise = torch.randn(x01.shape, generator=generator, device="cpu").to(x01.device)
        return (x01 + s * noise).clamp(0, 1)
    if kind == "salt_pepper":
        return corrupt(x01, "salt_pepper", s)
    if kind == "occlusion":
        return corrupt(x01, "mask", s)
    if kind == "blur":
        return _gaussian_blur(x01, s).clamp(0, 1)
    if kind == "contrast":
        mean = x01.mean(dim=(1, 2, 3), keepdim=True)
        return ((x01 - mean) * s + mean).clamp(0, 1)
    if kind == "brightness":
        return (x01 + s).clamp(0, 1)
    raise ValueError(f"Unknown test corruption {kind!r}. Choose from {TEST_CORRUPTIONS}")
