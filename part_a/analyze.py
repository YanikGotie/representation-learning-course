"""Figures, tables and extra evidence for the Part A report (mode "analyze", new in Part A).

Each function receives the config module as `cfg` (ANALYSES and ENCODERS come from the run's config file).

Each entry of `analyses` in the config is one function below. Outputs go to results/figures/
(PNG + a CSV/Markdown with the numbers behind every figure, so no claim rests on a picture).

  tuning          A2  validation reconstruction MSE vs k-NN accuracy over all trials (+ Spearman rho)
  table           A3  main results table (test top-1 / top-5) from results/protocol.csv
  label_efficiency A4 accuracy vs label fraction, DAE vs supervised-from-scratch (and AE, linear probe)
  rank_agreement  A5  do k-NN, linear probe and fine-tuning rank the models the same way? (Kendall tau)
  layer           A2  which layer is h: k-NN accuracy of the bottleneck vs the last conv map
  tsne            A5  t-SNE of validation features, 10 classes, per encoder
  retrieval       A5  nearest-neighbour retrieval grid (DAE vs raw pixels vs supervised)
  per_class       A5  best / worst classes, and DAE-vs-supervised per-class scatter
  reconstructions A1  clean / corrupted / reconstructed examples (DAE and AE)
  robustness      A4 bonus: accuracy under test-time corruptions of increasing severity
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

import config  # noqa: E402
from evaluation import (extract_features, knn_predict, per_class_accuracy, probe_predict, topk_acc,  # noqa: E402
                        train_linear_probe)
from augmentations import TEST_CORRUPTIONS, corrupt, test_corruption, to_float  # noqa: E402
from model import AutoEncoder, load_encoder  # noqa: E402

# Fixed categorical order (validated: CVD + normal-vision separation pass on a light surface).
# Colour follows the model, never its rank; every series also gets its own marker.
STYLE = {
    "dae": ("#2a78d6", "o", "DAE (ours)"),
    "ae": ("#eb6834", "s", "Vanilla AE"),
    "supervised": ("#1baf7a", "^", "Supervised"),
    "random": ("#eda100", "D", "Random init"),
    "pixels": ("#e87ba4", "v", "Raw pixels"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DATASET_NAMES = {"tiny_imagenet": "Tiny-ImageNet", "cifar100": "CIFAR-100 (transfer)"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    "lines.markersize": 6,
})


def style(tag):
    return STYLE.get(tag, ("#6250d6", "P", tag))


def fig_dir(cfg):
    d = os.path.join(cfg.RESULTS_DIR, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def _save(fig, cfg, name):
    path = os.path.join(fig_dir(cfg), name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def _protocol(cfg, all_layers=False):
    """results/protocol.csv; by default only rows on h (tags without '@<layer>')."""
    df = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "protocol.csv"))
    return df if all_layers else df[~df.tag.str.contains("@")].copy()


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tagged_encoders(cfg):
    """cfg.ENCODERS: {tag: path | 'random' | 'pixels'} -> {tag: encoder on device, eval mode}."""
    out = {}
    for tag, path in cfg.ENCODERS.items():
        pl.seed_everything(cfg.SEED, verbose=False)
        enc = load_encoder(path, cfg.LATENT_DIM)  # the width of h is read from the file
        out[tag] = enc.to(_device()).eval()
    return out


def _dm(cfg, dataset="tiny_imagenet"):
    from train import make_datamodule
    return make_datamodule(dataset, batch_size=256)


def _class_words(names, data_dir="dataset/"):
    """WordNet ids -> readable names, from words.txt in the Tiny-ImageNet zip (first synonym only)."""
    import zipfile
    zpath = os.path.join(data_dir, "tiny-imagenet-200.zip")
    try:
        with zipfile.ZipFile(zpath) as z:
            lines = z.read("tiny-imagenet-200/words.txt").decode().splitlines()
    except (OSError, KeyError):
        return list(names)
    table = dict(line.split("\t", 1) for line in lines if "\t" in line)
    return [table.get(n, n).split(",")[0].strip() for n in names]


# --------------------------------------------------------------------------------------
# A2: reconstruction vs representation
# --------------------------------------------------------------------------------------
def tuning(cfg, ctx):
    runs = pd.read_csv(os.path.join(cfg.RESULTS_DIR, "pretrain_runs.csv"))
    t = runs[runs.run_name.str.startswith("tune/")].copy()
    if t.empty:
        print("  no tuning trials yet")
        return
    rho, p = stats.spearmanr(t.val_recon_mse, t.val_knn_top1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {"gaussian": "#2a78d6", "mask": "#eb6834", "salt_pepper": "#1baf7a", "none": "#eda100"}
    markers = {"gaussian": "o", "mask": "s", "salt_pepper": "^", "none": "D"}
    for corr, g in t.groupby("corruption"):
        axes[0].scatter(g.val_recon_mse, 100 * g.val_knn_top1, s=10 + g.latent_dim / 8, c=colors.get(corr, MUTED),
                        marker=markers.get(corr, "o"), edgecolors="white", linewidths=0.8, label=corr, zorder=3)
        if corr != "none":
            axes[1].scatter(g.noise_level, 100 * g.val_knn_top1, c=colors.get(corr, MUTED),
                            marker=markers.get(corr, "o"), edgecolors="white", linewidths=0.8, label=corr, zorder=3)
    dae_trials = t[t.corruption != "none"]
    if not dae_trials.empty:  # the trial tune.py selected as the DAE
        best = dae_trials.loc[dae_trials.val_knn_top1.idxmax()]
        axes[0].annotate("selected DAE", (best.val_recon_mse, 100 * best.val_knn_top1), xytext=(8, -12),
                         textcoords="offset points", color=INK, fontsize=8,
                         arrowprops=dict(arrowstyle="-", color=MUTED))
    axes[0].set(xlabel="Validation reconstruction MSE (clean input, lower = better)",
                ylabel="Validation k-NN top-1 (%)",
                title=f"Reconstruction vs representation\nSpearman ρ = {rho:.2f} (p = {p:.2g}, {len(t)} trials)")
    axes[0].legend(title="corruption (size = latent dim)", fontsize=7, title_fontsize=7)
    axes[1].set(xlabel="Noise level (σ / mask ratio / s&p fraction)", ylabel="Validation k-NN top-1 (%)",
                title="k-NN accuracy vs\ncorruption strength")
    axes[1].legend(fontsize=7)
    _save(fig, cfg, "a2_recon_vs_knn.png")
    cols = ["run_name", "corruption", "noise_level", "latent_dim", "learning_rate", "weight_decay", "val_recon_mse",
            "val_denoise_mse", "val_knn_top1", "val_knn_top5", "best_epoch"]
    t[cols].sort_values("val_knn_top1", ascending=False).to_csv(os.path.join(fig_dir(cfg), "a2_trials.csv"),
                                                                index=False)
    ctx["tuning_spearman"] = {"rho": rho, "p": p, "n_trials": len(t)}


# --------------------------------------------------------------------------------------
# A3: main table
# --------------------------------------------------------------------------------------
def table(cfg, ctx):
    df = _protocol(cfg, all_layers=True)
    df["setting"] = df.apply(lambda r: r.protocol if r.protocol == "knn"
                             else f"{r.protocol}@{int(round(100 * r.label_fraction))}%", axis=1)
    order = ["knn", "linear_probe@1%", "linear_probe@10%", "linear_probe@100%",
             "finetune@1%", "finetune@10%", "finetune@100%"]
    df["layer"] = df.tag.str.split("@").str[1].fillna("h")
    df["tag"] = df.tag.str.split("@").str[0]
    lines = []
    for (lay, ds), g in df.groupby(["layer", "dataset"], sort=False):
        g = g.assign(cell=g.apply(lambda r: f"{100 * r.test_top1:.1f} / {100 * r.test_top5:.1f}", axis=1))
        piv = g.pivot_table(index="tag", columns="setting", values="cell", aggfunc="first")
        piv = piv[[c for c in order if c in piv.columns]]
        piv = piv.reindex([t for t in STYLE if t in piv.index] + [t for t in piv.index if t not in STYLE])
        where = "h = bottleneck" if lay == "h" else f"frozen features from the {lay} layer (pooled), not h"
        lines += [f"### {DATASET_NAMES.get(ds, ds)} ({where}): test top-1 / top-5 (%)", "",
                  piv.fillna("-").to_markdown(), ""]
    path = os.path.join(fig_dir(cfg), "a3_results_table.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"  wrote {path}")


def _end_labels(ax, ends, x=100):
    """Value labels at the right end of each line, spread out to a minimum gap, in the line's colour."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends = sorted(ends)
    pos = [ends[0][0]]
    for v, _ in ends[1:]:
        pos.append(max(v, pos[-1] + gap))
    for (v, color), y in zip(ends, pos):
        ax.annotate(f"{v:.1f}", (x, v), xytext=(x * 1.12, y), textcoords="data", va="center", fontsize=7,
                    color=color, fontweight="bold", annotation_clip=False)


# --------------------------------------------------------------------------------------
# A4: label efficiency
# --------------------------------------------------------------------------------------
def label_efficiency(cfg, ctx):
    df = _protocol(cfg)
    dsets = [d for d in DATASET_NAMES if d in df.dataset.unique()]
    fig, axes = plt.subplots(1, len(dsets), figsize=(4.4 * len(dsets), 3.4), squeeze=False)
    for ax, ds in zip(axes[0], dsets):
        g = df[df.dataset == ds]
        ends = []
        for tag in ["dae", "ae", "supervised"]:
            ft = g[(g.tag == tag) & (g.protocol == "finetune")].sort_values("label_fraction")
            if ft.empty:
                continue
            color, marker, label = style(tag)
            label = "Supervised from scratch" if tag == "supervised" else f"{label}, fine-tuned"
            ax.plot(100 * ft.label_fraction, 100 * ft.test_top1, color=color, marker=marker, label=label,
                    markeredgecolor="white", zorder=3)
            ends.append((100 * ft.test_top1.iloc[-1], color))
        lp = g[(g.tag == "dae") & (g.protocol == "linear_probe")].sort_values("label_fraction")
        if not lp.empty:
            ax.plot(100 * lp.label_fraction, 100 * lp.test_top1, color=style("dae")[0], marker="o", ls="--",
                    markerfacecolor="white", label="DAE (ours), linear probe", zorder=3)
        _end_labels(ax, ends)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100], ["1%", "10%", "100%"])
        ax.set(xlabel="Labelled training data", ylabel="Test top-1 (%)", title=DATASET_NAMES.get(ds, ds))
    axes[0][0].legend(fontsize=7)
    _save(fig, cfg, "a4_label_efficiency.png")


# --------------------------------------------------------------------------------------
# A5: rank agreement between protocols
# --------------------------------------------------------------------------------------
def rank_agreement(cfg, ctx):
    df = _protocol(cfg)
    rows = []
    for ds, g in df.groupby("dataset"):
        sel = {
            "knn": g[g.protocol == "knn"],
            "linear_probe@100%": g[(g.protocol == "linear_probe") & (g.label_fraction == 1.0)],
            "finetune@100%": g[(g.protocol == "finetune") & (g.label_fraction == 1.0)],
            "finetune@1%": g[(g.protocol == "finetune") & (g.label_fraction == 0.01)],
        }
        acc = pd.DataFrame({k: v.set_index("tag").test_top1 for k, v in sel.items()})
        acc.to_csv(os.path.join(fig_dir(cfg), f"a5_rank_{ds}.csv"))
        keys = list(sel)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                both = acc[[keys[i], keys[j]]].dropna()
                if len(both) >= 3:
                    tau, p = stats.kendalltau(both[keys[i]], both[keys[j]])
                    rows.append(dict(dataset=ds, a=keys[i], b=keys[j], n_models=len(both), kendall_tau=tau,
                                     ranking_a=" > ".join(both[keys[i]].sort_values(ascending=False).index),
                                     ranking_b=" > ".join(both[keys[j]].sort_values(ascending=False).index)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(fig_dir(cfg), "a5_rank_agreement.csv"), index=False)
    print(out.to_string())


# --------------------------------------------------------------------------------------
# A2: which layer is h
# --------------------------------------------------------------------------------------
def layer(cfg, ctx):
    dm = _dm(cfg)
    rows = []
    for tag, enc in ctx["encoders"].items():
        if tag == "pixels":
            continue
        for lay in ["conv", "h"]:
            bank = extract_features(enc, dm.bank_x, layer=lay)
            val = extract_features(enc, dm.val_x, layer=lay)
            s = knn_predict(bank, dm.bank_y, val, dm.num_classes, k=cfg.KNN_K)
            rows.append(dict(tag=tag, layer=lay, dim=bank.shape[1], val_knn_top1=topk_acc(s, dm.val_y, 1)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(fig_dir(cfg), "a2_layer_choice.csv"), index=False)
    print(out.to_string())
    piv = out.pivot(index="tag", columns="layer", values="val_knn_top1")
    fig, ax = plt.subplots(figsize=(4.6, 3))
    x = np.arange(len(piv))
    for i, (lay, hatch) in enumerate([("conv", "//"), ("h", "")]):
        ax.bar(x + (i - 0.5) * 0.38, 100 * piv[lay], width=0.36, hatch=hatch, edgecolor="white", linewidth=1.5,
               color=[style(t)[0] for t in piv.index],
               label="last conv map, pooled 4×4" if lay == "conv" else "bottleneck h (fc)")
    ax.set_xticks(x, [style(t)[2] for t in piv.index], fontsize=8)
    ax.set(ylabel="Validation k-NN top-1 (%)", title="Which layer is the representation?")
    ax.legend(fontsize=7)
    _save(fig, cfg, "a2_layer_choice.png")


# --------------------------------------------------------------------------------------
# A5: t-SNE and retrieval
# --------------------------------------------------------------------------------------
def tsne(cfg, ctx):
    from sklearn.manifold import TSNE

    dm = _dm(cfg)
    rng = np.random.default_rng(cfg.SEED)
    classes = np.sort(rng.choice(dm.num_classes, 10, replace=False))
    mask = np.isin(dm.val_y.numpy(), classes)
    x, y = dm.val_x[mask], dm.val_y[mask].numpy()
    names = _class_words([dm.class_names[c] for c in classes])
    tags = [t for t in ctx["encoders"]]
    fig, axes = plt.subplots(1, len(tags), figsize=(3.2 * len(tags), 3.3), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ax, tag in zip(axes[0], tags):
        f = extract_features(ctx["encoders"][tag], x).numpy()
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)  # same cosine geometry as k-NN
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=cfg.SEED).fit_transform(f)
        for i, c in enumerate(classes):
            ax.scatter(*emb[y == c].T, s=6, color=cmap(i), label=names[i])
        sil = _silhouette(f, y)
        ax.set_title(f"{style(tag)[2]}\nsilhouette (cosine) = {sil:.3f}", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    axes[0][-1].legend(fontsize=6, markerscale=2, loc="center left", bbox_to_anchor=(1, 0.5))
    _save(fig, cfg, "a5_tsne.png")


def _silhouette(f, y):
    from sklearn.metrics import silhouette_score
    return silhouette_score(f, y, metric="cosine")


def retrieval(cfg, ctx, n_query=6, n_nn=7):
    dm = _dm(cfg)
    rng = np.random.default_rng(cfg.SEED)
    q_idx = rng.choice(len(dm.test_y), n_query, replace=False)
    tags = [t for t in ["dae", "pixels", "supervised"] if t in ctx["encoders"]]
    fig, axes = plt.subplots(n_query * len(tags), n_nn + 1, figsize=(1.1 * (n_nn + 1), 1.15 * n_query * len(tags)))
    for ti, tag in enumerate(tags):
        enc = ctx["encoders"][tag]
        bank = torch.nn.functional.normalize(extract_features(enc, dm.bank_x), dim=1)
        q = torch.nn.functional.normalize(extract_features(enc, dm.test_x[q_idx]), dim=1)
        nn_idx = (q @ bank.T).topk(n_nn, dim=1).indices
        for qi in range(n_query):
            r = ti * n_query + qi
            imgs = [dm.test_x[q_idx[qi]]] + [dm.bank_x[j] for j in nn_idx[qi]]
            labels = [int(dm.test_y[q_idx[qi]])] + [int(dm.bank_y[j]) for j in nn_idx[qi]]
            for ci, (im, lab) in enumerate(zip(imgs, labels)):
                ax = axes[r, ci]
                ax.imshow(im.permute(1, 2, 0).numpy())
                ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(True)
                    s.set_linewidth(2.5 if ci > 0 else 1)
                    s.set_color(MUTED if ci == 0 else ("#0ca30c" if lab == labels[0] else "#d03b3b"))
            axes[r, 0].set_ylabel(style(tag)[2] if qi == 0 else "", fontsize=8)
    axes[0, 0].set_title("query", fontsize=8)
    for ci in range(1, n_nn + 1):
        axes[0, ci].set_title(f"NN {ci}", fontsize=8)
    fig.subplots_adjust(top=1 - 0.35 / fig.get_figheight(), hspace=0.12, wspace=0.06)
    fig.suptitle("Nearest-neighbour retrieval on test queries (green border = same class, red = different)",
                 fontsize=9, y=1 - 0.08 / fig.get_figheight())
    _save(fig, cfg, "a5_retrieval.png")


# --------------------------------------------------------------------------------------
# A5: per-class failures
# --------------------------------------------------------------------------------------
def per_class(cfg, ctx):
    d = os.path.join(cfg.RESULTS_DIR, "per_class")
    dm = _dm(cfg)
    words = _class_words(dm.class_names, cfg.DATA_DIR)
    out = {}
    for proto in ["linear", "knn"]:
        path = os.path.join(d, f"dae_tiny_imagenet_{proto}.npy")
        if not os.path.exists(path):
            continue
        acc = np.load(path)
        order = np.argsort(acc)
        out[proto] = {"worst": [(words[i], float(acc[i])) for i in order[:10]],
                      "best": [(words[i], float(acc[i])) for i in order[::-1][:10]],
                      "zero_acc_classes": int((acc == 0).sum())}
        sup = os.path.join(d, f"supervised_tiny_imagenet_{proto}.npy")
        if proto == "linear" and os.path.exists(sup):
            s = np.load(sup)
            fig, ax = plt.subplots(figsize=(3.8, 3.6))
            ax.scatter(100 * s, 100 * acc, s=10, color=style("dae")[0], edgecolors="white", linewidths=0.5)
            ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--")
            r, _ = stats.pearsonr(s, acc)
            for i in order[::-1][:4]:  # best classes only: the worst all sit at (x, 0) and would overlap
                ax.annotate(words[i], (100 * s[i], 100 * acc[i]), fontsize=6, color=INK, xytext=(3, 3),
                            textcoords="offset points")
            ax.set(xlabel="Supervised encoder, per-class acc (%)", ylabel="DAE, per-class acc (%)",
                   title=f"Per-class linear-probe accuracy (r = {r:.2f})")
            _save(fig, cfg, "a5_per_class_scatter.png")
    with open(os.path.join(fig_dir(cfg), "a5_per_class.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


# --------------------------------------------------------------------------------------
# A1: reconstructions
# --------------------------------------------------------------------------------------
def reconstructions(cfg, ctx, n=8):
    dm = _dm(cfg)
    rows, labels = [], []
    x01 = to_float(dm.val_x[:n]).to(_device())
    rows.append(x01), labels.append("clean x")
    for tag in ["dae", "ae"]:
        path = cfg.ENCODERS.get(tag)
        ckpt = os.path.join(os.path.dirname(path or ""), "best.ckpt")
        if not path or not os.path.exists(ckpt):
            continue
        m = AutoEncoder.load_from_checkpoint(ckpt, map_location="cpu").to(_device()).eval()
        torch.manual_seed(cfg.SEED)
        xt = corrupt(x01, m.hparams.corruption, m.hparams.noise_level, m.hparams.mask_patch)
        with torch.no_grad():
            if tag == "dae":
                rows.append(xt), labels.append(f"x~ ({m.hparams.corruption}, {m.hparams.noise_level:.2f})")
                rows.append(m(xt).clamp(0, 1)), labels.append("DAE g(f(x~))")
            rows.append(m(x01).clamp(0, 1)), labels.append(f"{style(tag)[2]} g(f(x))")
    fig, axes = plt.subplots(len(rows), n, figsize=(1.0 * n, 1.05 * len(rows)))
    for r, (imgs, lab) in enumerate(zip(rows, labels)):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c].permute(1, 2, 0).float().cpu().numpy())
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        axes[r, 0].set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
    _save(fig, cfg, "a1_reconstructions.png")


# --------------------------------------------------------------------------------------
# A4 bonus: robustness to test-time corruptions
# --------------------------------------------------------------------------------------
def robustness(cfg, ctx):
    """Train each linear probe on CLEAN features, then evaluate on corrupted test images.

    Probes whether h is stable under low-level shifts. Fair: every encoder sees the same
    corrupted images (fixed seed), the classifier never sees corruptions, and the test set
    includes shifts the DAE never trained on (blur, contrast, brightness) next to the ones
    it did (noise, salt & pepper, occlusion), so in- and out-of-training shifts can be compared.
    """
    dm = _dm(cfg)
    rows = []
    for tag, enc in ctx["encoders"].items():
        bank_f = extract_features(enc, dm.bank_x)
        val_f = extract_features(enc, dm.val_x)
        probe, _, _ = train_linear_probe(bank_f, dm.bank_y, val_f, dm.val_y, dm.num_classes)
        for kind in TEST_CORRUPTIONS:
            for sev in cfg.ROBUSTNESS_SEVERITIES:
                f = extract_features(enc, dm.test_x, corruption=kind, severity=sev, seed=cfg.SEED)
                lin = probe_predict(probe, f)
                knn = knn_predict(bank_f, dm.bank_y, f, dm.num_classes, k=cfg.KNN_K)
                rows.append(dict(tag=tag, corruption=kind, severity=sev, linear_top1=topk_acc(lin, dm.test_y, 1),
                                 knn_top1=topk_acc(knn, dm.test_y, 1)))
        print(f"  robustness done: {tag}")
    df = pd.DataFrame(rows)
    clean = df[df.severity == 0].groupby("tag")[["linear_top1", "knn_top1"]].first()
    df["linear_rel"] = df.apply(lambda r: r.linear_top1 / clean.loc[r.tag, "linear_top1"], axis=1)
    df["knn_rel"] = df.apply(lambda r: r.knn_top1 / clean.loc[r.tag, "knn_top1"], axis=1)
    df.to_csv(os.path.join(fig_dir(cfg), "a4_robustness.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, kind in zip(axes.flat, TEST_CORRUPTIONS):
        g = df[df.corruption == kind]
        for tag in ctx["encoders"]:
            gt = g[g.tag == tag]
            color, marker, label = style(tag)
            ax.plot(gt.severity, 100 * gt.linear_rel, color=color, marker=marker, label=label,
                    markeredgecolor="white", zorder=3)
        ax.set_title(kind.replace("_", " "))
        ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    for ax in axes[1]:
        ax.set_xlabel("Severity (0 = clean)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Linear probe acc,\n% of clean acc")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Robustness: linear probe trained on clean features, tested on corrupted Tiny-ImageNet test images",
                 fontsize=10)
    _save(fig, cfg, "a4_robustness.png")
    summary = df[df.severity > 0].groupby(["tag", "corruption"]).linear_rel.mean().unstack()
    summary.to_csv(os.path.join(fig_dir(cfg), "a4_robustness_summary.csv"))
    print(summary.round(3).to_string())


ANALYSES = dict(tuning=tuning, table=table, label_efficiency=label_efficiency, rank_agreement=rank_agreement,
                layer=layer, tsne=tsne, retrieval=retrieval, per_class=per_class, reconstructions=reconstructions,
                robustness=robustness)
NEEDS_ENCODERS = {"layer", "tsne", "retrieval", "robustness"}


def run_analyze(cfg=config):
    ctx = {}
    if NEEDS_ENCODERS & set(cfg.ANALYSES):
        ctx["encoders"] = load_tagged_encoders(cfg)
    for name in cfg.ANALYSES:
        print(f"== {name}")
        ANALYSES[name](cfg, ctx)
    if ctx.get("tuning_spearman"):
        print("tuning:", ctx["tuning_spearman"])
