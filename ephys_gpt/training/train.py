from __future__ import annotations

import pytorch_lightning as pl
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from torch.utils.data import IterableDataset
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torchmetrics import F1Score, ConfusionMatrix
import os
import torchview  # noqa: F401  # optional, used only for model graph plotting
import matplotlib.pyplot as plt
import seaborn as sns

from ..dataset import split_datasets
from .lightning import LitModel
from .lightning import DatasetEpochCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from .utils import get_model_class, get_loss_class


class ExperimentTokenizer:
    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: Configuration dictionary
        """
        datasets = split_datasets(**cfg["datasplitter"])

        # get all training data
        train_data = []
        for i in range(len(datasets.train)):
            x, _ = datasets.train[i]
            train_data.append(x[0])

        train_data = torch.stack(train_data).permute(0, 2, 1)  # (B, T, C)
        print(train_data.shape)

        tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        tokenizer = tokenizer.fit(action_data=train_data, vocab_size=cfg["vocab_size"])

        tokenizer.save_pretrained(cfg["save_dir"])


class ExperimentDL:
    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: Configuration dictionary
        """
        # print pytorch version
        print("--------------------------------")
        print(f"PyTorch Lightning version: {pl.__version__}")
        print(f"PyTorch version: {torch.__version__}")
        print("--------------------------------")

        self.trainer_args = cfg["trainer"]
        self.dataloader_args = cfg["dataloader"]
        self.resume_from = cfg["resume_from"]
        self.dataset_name = cfg.get("dataset_name", "omega")
        self.save_dir = cfg["save_dir"]

        self.trainer_args["default_root_dir"] = cfg["save_dir"]

        # load model and augmentations
        with open(cfg["model_config"]) as f:
            model_cfg = yaml.safe_load(f)

        if cfg.get("dataset_name", "omega") == "omega":
            self.datasets = split_datasets(**cfg["datasplitter"])
        else:
            raise ValueError(f"Invalid dataset name: {cfg['dataset_name']}")

        # Get model and loss classes dynamically
        model_class = get_model_class(cfg["model_name"])
        loss_class = get_loss_class(cfg["loss_name"])

        postprocessor = getattr(self.datasets.train, "postprocessor", None)

        self.lit_model = LitModel(
            model_class=model_class,
            loss_class=loss_class,
            model_cfg=model_cfg,
            loss_cfg=cfg["loss"],
            trainer_cfg=cfg["lightning"],
            postprocessor=postprocessor,
        )

        best_ckpt = ModelCheckpoint(
            monitor="val_loss",  # metric to monitor
            mode="min",  # 'min' for loss, 'max' for accuracy or similar
            save_top_k=1,  # save only the best model
            filename="best-checkpoint",  # optional: custom filename
        )
        epoch_ckpt = ModelCheckpoint(
            filename="last-checkpoint",
            save_top_k=-1,                 # keep every epoch
            every_n_train_steps=self.trainer_args.get("log_every_n_steps", 100),
            save_on_train_epoch_end=True,  # trigger right after each epoch
        )

        callbacks = self.trainer_args.get("callbacks", []) or []
        callbacks.extend([best_ckpt, epoch_ckpt])

        # If the training dataset exposes an epoch hook, add a callback
        if hasattr(self.datasets, "train") and (
            hasattr(self.datasets.train, "set_epoch")
            or hasattr(self.datasets.train, "on_epoch_start")
        ):
            callbacks.append(DatasetEpochCallback(self.datasets.train))
            callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

        self.trainer_args["callbacks"] = callbacks

    def _visualize_model(self, cfg: dict) -> None:
        # Use torchview to visualize the model
        x = torch.randn(1, 2, 100)
        sensor_pos_ori = torch.randn(1, 2, 2)
        sensor_type = torch.randint(0, 2, (1, 2))
        self.model_graph = torchview.draw_graph(
            self.lit_model.model,
            input_data=[(x, sensor_pos_ori, sensor_type)],
            save_graph=True,
            directory=cfg["save_dir"],
            filename="model_graph.pdf",
        )

    def train(self) -> None:
        args = self.dataloader_args

        shuffle = None if isinstance(self.datasets.train, IterableDataset) else True
        train_loader = DataLoader(self.datasets.train, shuffle=shuffle, **args)

        shuffle = None if isinstance(self.datasets.val, IterableDataset) else False
        val_loader = DataLoader(self.datasets.val, shuffle=shuffle, **args)
        trainer = pl.Trainer(**self.trainer_args)
        trainer.fit(
            self.lit_model, train_loader, val_loader, ckpt_path=self.resume_from
        )

    def test(self) -> None:
        args = self.dataloader_args
        test_loader = DataLoader(self.datasets.test, shuffle=False, **args)
        trainer = pl.Trainer(**self.trainer_args)
        trainer.test(self.lit_model, test_loader, ckpt_path=self.resume_from)

        if self.lit_model.test_targets:
            # compute F1 score and confusion matrix
            targets = torch.cat(self.lit_model.test_targets)
            preds = torch.cat(self.lit_model.test_predictions)
            preds_classes = preds.argmax(dim=-1)
            num_classes = preds.size(-1)

            f1_macro = F1Score(
                task="multiclass", average="macro", num_classes=num_classes
            )
            f1_score = f1_macro(preds, targets)
            print(f"F1 score: {f1_score}")

            # Compute confusion matrix
            cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)
            confusion = cm(preds_classes, targets)
            confusion_np = confusion.cpu().numpy()

            # Save confusion matrix to file
            os.makedirs(self.save_dir, exist_ok=True)
            cm_pdf_path = os.path.join(self.save_dir, "confusion_matrix.pdf")

            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_np, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(cm_pdf_path)
            plt.close()
            print(f"Confusion matrix saved to {cm_pdf_path}")
            return

        # save test predictions
        if self.dataset_name == "libribrain":
            # flatten list of batched predictions
            preds = torch.cat(self.lit_model.test_predictions)
            preds = [p for p in preds]

            self.datasets.test.generate_submission_in_csv(
                preds, f"{self.save_dir}/holdout_phoneme_predictions.csv"
            )
