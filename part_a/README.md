# Part A: Denoising-autoencoder representations on Tiny-ImageNet

`part_a/` follows the course convention: it started as a copy of `lec5/` and then changed.
It keeps lec5's layout and pipeline (`config.py`, `train.py`, `utils.py` with `parse_args` /
`build_model`, `model.py`, `datamodule.py`, `callbacks.py`, `augmentations.py`, `losses.py`).
Every method is a `LightningModule`, the data is a `LightningDataModule`, and **every run is
`train.py` with its own config file** in `configs/`.

```bash
cd part_a
pip install -r requirements.txt
python train.py --config configs/<run>.yaml                                  # one run
python train.py --config configs/<run>.yaml --set NUM_EPOCHS=2 TRAIN_PER_CLASS=20   # quick check
bash run_all.sh                                                             # every Part A run, in order
tensorboard --logdir tb_logs
```

A run's YAML file overrides the defaults in `config.py` (same UPPERCASE names). `--mode`,
`--dataset` and `--set NAME=VALUE` override the file. The merged values are saved to
`checkpoints/<RUN_NAME>/config.yaml`. On Colab, [`colab_train.ipynb`](colab_train.ipynb) (the course's notebook, pointed at this fork and `part_a/`)
only clones the repo, installs it and runs `train.py`.

## What changed from lec5

| lec5 | part_a | Why |
|---|---|---|
| `MnistDataModule`, 28×28 grey | `TinyImageNetDataModule`: Tiny-ImageNet and CIFAR-100, 64×64 RGB, cached once as uint8 | The assignment's data. Whole batches go to the GPU; per-image CPU transforms would be the bottleneck on Colab's 2 cores. |
| Random 50k/10k split, first `N_TARGET_TRAIN` images | 50 images/class held out as validation; official val = test; class-balanced 1/10/100% label subsets, seeded | Identical, balanced splits and subsets for every model. |
| Encoder written out in every class (1 channel, 7×7) | `build_encoder()`: the same layers for 3 channels, 64→32→16, `Linear(16384→d)` | One definition, so an encoder file saved by one model loads into any other. |
| `AutoEncoder` | `AutoEncoder` with a corruption q(x̃\|x) (Gaussian, block mask, salt & pepper; `"none"` = lec5's plain AE); loss on [0,1] pixels; AdamW + cosine | A1, the DAE; the plain AE is its counterpart. |
| `EncoderClassifier` (`linear_probe`, `finetune`) | Same class. The linear probe trains on features the frozen encoder computed once; fine-tuning adds crop + flip, warm-up + cosine; top-1 and top-5 | A3 protocol. |
| `SupervisedEncoder` | An `EncoderClassifier` with no pretrained file, trained with the fine-tuning recipe | The fair "supervised from scratch" baseline. |
| `EarlyStopping` on `val_loss` | Off by default (`EARLY_STOPPING_PATIENCE`); `ModelCheckpoint` keeps the best validation epoch | Cosine schedules assume the full number of epochs; the DAE is selected by k-NN, not by its loss. |
| — | `callbacks.KNNMonitor`, `evaluation.py` (k-NN, linear probe), `tune.py` (Optuna), `analyze.py` (figures) | k-NN protocol, A2 tuning, report figures. |
| `SimCLR`, `MoCo`, `losses.py` | Kept, on the 64×64 RGB encoder; their modes return in Part B | Part B builds on the same folder. |

## Modes (`MODEL_NAME` in the config file, or `--mode`)

| Mode | Model | Does |
|---|---|---|
| `pretrain` | `AutoEncoder` | Trains the (denoising) AE on unlabelled Tiny-ImageNet, keeps the epoch with the best validation k-NN, saves the encoder to `ENCODER_PATH`. |
| `pretrain_supervise_encoder` | `SupervisedEncoder` | Supervised reference encoder on 100% of the labels; saves the encoder. |
| `knn`, `linear_probe`, `finetune`, `evaluate` | `EncoderClassifier` | One protocol (or all of `PROTOCOLS`) for the encoder in `ENCODER_PATH`, on every dataset in `EVAL_DATASETS` → `results/protocol.csv`. `ENCODER_PATH: random` gives the random-init / from-scratch baselines, `pixels` the raw-pixel baseline. |
| `tune` | `AutoEncoder` | A2: Optuna search, writes `results/best_configs/`. |
| `analyze` | — | Figures and tables → `results/figures/`. |

## Setup (common to Parts A and B)

| Item | Setting |
|---|---|
| Pre-training data | Tiny-ImageNet train (labels never reach the loss). |
| Splits | **train** 90k (450/class) · **val** 10k = 50/class held out of train, used for *all* tuning and selection · **test** = official labelled val (10k), final numbers only. Seed 42. |
| Label subsets | 1% / 10% / 100% of train, class-balanced (5 / 45 / 450 per class), same seed → identical for every model. CIFAR-100: 5 / 45 / 450 per class of its 45k train split. |
| Encoder f_θ | lec5's small encoder for RGB 64×64: `Conv(3→32,s2)–ReLU–Conv(32→64,s2)–ReLU–Flatten–Linear(16384→d)` |
| Representation h | The bottleneck output (`Linear`, d-dim), as the course encoder defines it. `analyses: layer` and the `evaluate_*_conv` runs compare it with the last conv map average-pooled to 4×4 (1024-d). |
| Decoder g_φ | `Linear(d→16384)–ReLU–ConvT(64→32,s2)–ReLU–ConvT(32→3,s2)`, linear output in [0,1] pixel space. |
| DAE loss | `L = E_x E_{x̃~q(·|x)} ‖x − g(f(x̃))‖²`, x in [0,1]; corruptions: Gaussian noise, 8×8 block masking, salt & pepper. |
| Optimizer | AdamW + cosine schedule, batch 256, `precision="16-mixed"`. |
| k-NN | ℓ2-normalized h, cosine similarity, k = 20, each neighbour votes with its similarity. Memory bank = full train split. |
| Linear probe | Frozen f_θ, features extracted once and standardized with train statistics; `EncoderClassifier`'s head trained with CE (AdamW, 30 epochs, ≥1500 steps); weight decay ∈ {0, 1e-4, 1e-3} chosen on val. |
| Fine-tuning | All weights, linear head, AdamW lr 1e-3, wd 5e-4, cosine with 1 warm-up epoch, crop+flip; epochs 100 / 40 / 20 for 1 / 10 / 100%; epoch chosen on val top-1. |
| Transfer | CIFAR-100 resized to 64×64 (bilinear, antialiased), normalized with Tiny-ImageNet statistics. |
| Metrics | top-1 and top-5 for every protocol. |

## Run order (`run_all.sh`)

| # | Config | Answers |
|---|---|---|
| 1 | `configs/tune_dae.yaml` | **A2**: 30 Optuna trials, proxy setting (150 img/class, 15 epochs). Writes `results/best_configs/`. |
| 2 | copy `results/best_configs/pretrain_{dae,ae}.yaml` over `configs/` (and commit) | |
| 3 | `configs/pretrain_dae.yaml`, `configs/pretrain_ae.yaml` | **A1**: final DAE and its plain counterpart, full setting (40 epochs). |
| 4 | `configs/supervised.yaml` | Supervised reference encoder (100% labels). |
| 5 | `configs/evaluate_{dae,ae,random,scratch,supervised,pixels}.yaml` | **A3**: full protocol on Tiny-ImageNet + CIFAR-100 → `results/protocol.csv`. |
| 5b | `configs/evaluate_{dae,ae,random,supervised}_conv.yaml` | **A2/A5**: k-NN and linear probe on the pooled conv map (`FEATURE_LAYER: conv`), rows tagged `<model>@conv`. |
| 6 | `configs/analyze.yaml` | **A2/A4/A5** figures and tables → `results/figures/`. |

Baselines (A3): `random` = random-init encoder, frozen · `ae` = plain counterpart · `supervised` =
`SupervisedEncoder` trained from scratch on the same label subsets (`evaluate_scratch`) and, frozen,
the encoder trained on 100% labels (`evaluate_supervised`) · `pixels` = raw 16×16 pixels (extra sanity baseline).

### Outputs → report sections

| Output | Section |
|---|---|
| `results/figures/a2_recon_vs_knn.png`, `a2_trials.csv` | A2 required plot: val reconstruction MSE vs val k-NN over all trials, Spearman ρ |
| `results/figures/a2_layer_choice.{png,csv}` | A2: which layer is h |
| `results/figures/a1_reconstructions.png` | A1: x, x̃, g(f(x̃)), and the AE's g(f(x)) |
| `results/figures/a3_results_table.md` | A3: all models × protocols × datasets, test top-1 / top-5 (on h; conv-feature tables below) |
| `results/figures/a4_label_efficiency.png` | A4: accuracy vs label fraction, DAE vs supervised |
| `results/figures/a4_robustness.{png,csv}`, `a4_robustness_summary.csv` | A4 bonus: test-time corruptions, severity 1–5 |
| `results/figures/a5_rank_agreement.csv` | A5: do k-NN, linear probe and fine-tuning rank models the same way (Kendall τ)? |
| `results/figures/a5_tsne.png`, `a5_retrieval.png`, `a5_per_class*.{png,json}` | A5: t-SNE + silhouette, retrieval grid, best / worst classes |
| `report/part_a_report.pdf` | The 3-page report (source: `report/part_a_report.html`) |

## Reproducibility

* Fixed seed (42) for splits, label subsets, initialization, data order and Optuna's sampler.
* Each run saves its merged config (`checkpoints/<run>/config.yaml`); pretraining also saves its per-epoch k-NN history (`history.json`).
* Console logs of every run are in `logs/`. A run writes its TensorBoard logs (losses, k-NN accuracy per epoch, reconstruction grids, learning rate) to `tb_logs/`, which is gitignored because the image grids make it 200+ MB. The scalars of every run behind the report are committed in `logs/tensorboard/`: `tensorboard --logdir logs/tensorboard`.
* Datasets and checkpoints are not committed (`.gitignore`). `results/` (CSVs, figures) and `logs/` are.

### Data source note

The assignment names Hugging Face (`zh-plus/tiny-imagenet`) as the source. huggingface.co is blocked on the network
these experiments ran on, so this project reads the **same 100k / 10k images** from the dataset's original host
(`tiny-imagenet-200.zip`, cs231n.stanford.edu); `DATA_SOURCE = "stanford"` is the default. Labels follow sorted WordNet
ids, and the official `val` folder is the test set. `--set DATA_SOURCE=hf` switches to Hugging Face. Use **one** source
for a whole project: the seeded split depends on the image order.

All runs here were done locally on an Apple M1 Pro (PyTorch MPS, `16-mixed`).
