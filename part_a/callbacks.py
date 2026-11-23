import torch
from pytorch_lightning.callbacks import EarlyStopping, Callback, TQDMProgressBar

from evaluation import knn_eval


class ColabProgressBar(TQDMProgressBar):
    """tqdm bar without the separate validation bar.

    The validation bar is drawn on a second line, and Colab can't redraw a second
    line in place, so every update would be printed as a new line.
    Validation metrics still appear at the end of the main epoch bar.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class EpochMetricsCallback(Callback):
    """Logs one point per epoch under "epoch/...", with the epoch number as the x-axis.

    Lightning's own charts use the training step as the x-axis. Pretraining uses the
    full training set; the target runs use fewer labeled images, so their curves
    don't line up. With the epoch as the x-axis, all modes can be compared on one chart.
    """

    def _log_metrics(self, trainer, names):
        metrics = {}
        for name in names:
            # with on_step=True Lightning stores the epoch value as "<name>_epoch"
            value = trainer.callback_metrics.get(f"{name}_epoch", trainer.callback_metrics.get(name))
            if value is not None:
                metrics[f"epoch/{name}"] = value.item()

        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_metrics(
            trainer,
            [
                "train_loss",
                "train_accuracy",
                "train_f1_score",
                "train_pos_sim",
                "train_neg_sim",
                "train_acc_top1",
                "train_acc_top5",
                "train_acc_mean_pos",
                "train_active",
            ],
        )

    def on_validation_end(self, trainer, pl_module):
        # Skip the sanity check and the extra trainer.validate() after training
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        self._log_metrics(
            trainer,
            [
                "val_loss",
                "val_accuracy",
                "val_top5",
                "val_recon_mse",
                "val_knn_top1",
                "val_knn_top5",
                "val_pos_sim",
                "val_neg_sim",
                "val_acc_top1",
                "val_acc_top5",
                "val_acc_mean_pos",
                "val_active",
            ],
        )


class KNNMonitor(Callback):
    """k-NN accuracy of h on the validation split, after every pretraining epoch.

    The assignment asks to choose the autoencoder by representation quality, not by its
    reconstruction loss. So ModelCheckpoint and the Optuna search both read val_knn_top1.
    The memory bank is the training split. In the short tuning runs it can be a class-balanced
    subset (bank_per_class) to save time. Labels are only used here to score h; they never
    reach the loss.
    """

    def __init__(self, datamodule, every_n_epochs=1, k=20, bank_per_class=None):
        super().__init__()
        self.dm = datamodule
        self.every = every_n_epochs
        self.k = k
        self.bank_per_class = bank_per_class
        self.history = []  # one row per epoch, saved to history.json

    def _bank(self):
        x, y = self.dm.bank_x, self.dm.bank_y
        if self.bank_per_class is None:
            return x, y
        g = torch.Generator().manual_seed(0)
        keep = torch.cat([
            (y == c).nonzero().view(-1)[torch.randperm(int((y == c).sum()), generator=g)[: self.bank_per_class]]
            for c in range(self.dm.num_classes)
        ])
        return x[keep], y[keep]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip the sanity check and the extra trainer.validate() after training
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        last = trainer.current_epoch + 1 >= trainer.max_epochs
        if (trainer.current_epoch + 1) % self.every != 0 and not last:
            return
        bank_x, bank_y = self._bank()
        res = knn_eval(pl_module.encoder, bank_x, bank_y, self.dm.val_x, self.dm.val_y, self.dm.num_classes, k=self.k)
        pl_module.log("val_knn_top1", res["top1"], prog_bar=True)
        pl_module.log("val_knn_top5", res["top5"])
        self.history.append({"epoch": trainer.current_epoch, "val_knn_top1": res["top1"], "val_knn_top5": res["top5"],
                             **{k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val_")}})
