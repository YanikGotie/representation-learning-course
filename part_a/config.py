# Part A: autoencoder representations on Tiny-ImageNet. Started as a copy of lec5.
# This file holds the defaults. Every run also has its own small YAML file in configs/,
# passed with --config, which overrides any name below. So each run is one file:
#   python train.py --config configs/pretrain_dae.yaml
# The merged values are saved next to the run's checkpoints (config.yaml), so every
# number in the report can be traced back to one file.

# Training hyperparameters
NUM_CLASSES = 200  # set from the dataset when a run starts (Tiny-ImageNet 200, CIFAR-100 100)
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001  # AdamW. lec5 used Adam; we tune weight decay, so the optimizer needs it.
BATCH_SIZE = 256
NUM_EPOCHS = 40
LATENT_DIM = 256  # length of h. Every encoder-based model must use the same value.
# Baselines without an encoder file (random init, supervised from scratch) copy the width of h
# from this file, so they always match the tuned DAE. None = use LATENT_DIM.
LATENT_DIM_FROM = None
SEED = 42  # pl.seed_everything, the hold-out split, the label subsets, and Optuna

# Dataset
DATA_DIR = "dataset/"
NUM_WORKERS = 0  # batches are sliced from in-memory tensors, so workers would only add overhead
# Tiny-ImageNet source: "hf" = zh-plus/tiny-imagenet (the assignment's link) or "stanford" = the
# original zip with the same images. huggingface.co was blocked on the network these runs used.
# Use one source for a whole project: the seeded hold-out split depends on the image order.
DATA_SOURCE = "stanford"
TRAIN_PER_CLASS = None  # proxy setting for tuning: a class-balanced subset of the train split

# Compute related
ACCELERATOR = "auto"  # GPU on Colab, MPS on an Apple-silicon Mac
DEVICES = 1
PRECISION = "16-mixed"
EARLY_STOPPING_PATIENCE = None  # None = off. Our cosine schedules assume the full number of epochs.

# One model name per run. --mode (or MODEL_NAME in the run's config file) is one of these:
#   "pretrain"                    : AutoEncoder - a denoising AE (CORRUPTION); "none" is the plain AE. Saves the encoder.
#   "pretrain_supervise_encoder"  : SupervisedEncoder - small encoder + head from scratch on all labels. Saves the encoder.
#   "knn"                         : k-NN on the frozen encoder (no training)
#   "linear_probe"                : EncoderClassifier - that encoder FROZEN + a new head
#   "finetune"                    : EncoderClassifier - that encoder + a new head, all weights.
#                                   With ENCODER_PATH = "random" it is the SupervisedEncoder baseline from scratch.
#   "evaluate"                    : every protocol in PROTOCOLS for one encoder (knn, linear_probe, finetune)
#   "tune"                        : Optuna search over the AutoEncoder's hyperparameters (tune.py)
#   "analyze"                     : figures and tables for the report (analyze.py)
# SimCLR and MoCo are still in model.py from lec4/lec5. Part B brings back their modes.
PRETRAIN_MODES = ("pretrain", "pretrain_supervise_encoder")
SAVE_ENCODER_MODES = ("pretrain", "pretrain_supervise_encoder")
EVAL_MODES = ("knn", "linear_probe", "finetune", "evaluate")
MODEL_NAME = "pretrain"
RUN_NAME = "debug"  # checkpoints/<RUN_NAME>/ and tb_logs/<RUN_NAME>/
TAG = None  # model name in the results tables (dae, ae, random, supervised, pixels); default RUN_NAME

# Denoising autoencoder (model.AutoEncoder). The encoder sees x~ ~ q(x~|x), the decoder rebuilds x.
#   "gaussian": NOISE_LEVEL = noise std    "mask": NOISE_LEVEL = share of MASK_PATCH x MASK_PATCH blocks set to 0
#   "salt_pepper": NOISE_LEVEL = share of pixels set to 0 or 1    "none": no corruption (plain AE)
CORRUPTION = "gaussian"
NOISE_LEVEL = 0.2
MASK_PATCH = 8

# k-NN (evaluation.knn_eval): cosine similarity on l2-normalized h, similarity-weighted vote.
KNN_K = 20
KNN_EVERY_N_EPOCHS = 1  # callbacks.KNNMonitor: validation k-NN during pretraining selects the epoch
KNN_BANK_PER_CLASS = None  # None = the whole train split is the memory bank

# Evaluation protocol (modes knn / linear_probe / finetune / evaluate)
ENCODER_PATH = None  # encoder file from a pretrain run, "random" (random init) or "pixels" (raw pixels)
DATASET = "tiny_imagenet"  # pretraining data; --dataset overrides it
EVAL_DATASETS = ["tiny_imagenet", "cifar100"]  # CIFAR-100 is the transfer set, resized to 64x64
PROTOCOLS = ["knn", "linear_probe", "finetune"]
LABEL_FRACTIONS = [0.01, 0.1, 1.0]  # class-balanced subsets of train, identical for every model
FEATURE_LAYER = "h"  # frozen protocols only: "h" = bottleneck, "conv" = last conv map pooled to 4x4
PROBE_EPOCHS = 30
PROBE_LEARNING_RATE = 0.001
PROBE_BATCH_SIZE = 512
PROBE_WEIGHT_DECAYS = [0.0, 0.0001, 0.001]  # chosen on validation
PROBE_MIN_STEPS = 1500  # 1% of the labels is only a few batches per epoch, so train at least this long
FINETUNE_LEARNING_RATE = 0.001
FINETUNE_WEIGHT_DECAY = 0.0005
FINETUNE_EPOCHS = {0.01: 100, 0.1: 40, 1.0: 20}  # fewer labels -> more epochs, so every fraction trains a similar number of steps
FINETUNE_BATCH_SIZE = 256

# Tuning (mode "tune", tune.py)
N_TRIALS = 30
SEARCH_SPACE = {}

# Analysis (mode "analyze", analyze.py)
ENCODERS = {}  # {tag: encoder file, "random" or "pixels"}
ANALYSES = []
ROBUSTNESS_SEVERITIES = [0, 1, 2, 3, 4, 5]

# Output folders
RESULTS_DIR = "results/"
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "tb_logs/"

# SimCLR / MoCo (lec4, lec5). Unused in Part A, kept for Part B.
TEMPERATURE = 0.07
TRIPLET_MARGIN = 0.2
PROJECTOR_DIM = 128
SIMCLR_BATCH_SIZE = 256
CONTRASTIVE_LOSS = "nt_xent"
MOCO_K = 4096
MOCO_M = 0.999
