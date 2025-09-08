from __future__ import annotations

import pytorch_lightning as pl
import yaml
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import torchview  # noqa: F401  # optional, used only for model graph plotting

from ..dataset import split_datasets, split_datasets_libribrain
from .lightning import LitModel

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
        self.trainer_args = cfg["trainer"]
        self.dataloader_args = cfg["dataloader"]
        self.resume_from = cfg["resume_from"]

        self.trainer_args["default_root_dir"] = cfg["save_dir"]

        # load model and augmentations
        with open(cfg["model_config"]) as f:
            model_cfg = yaml.safe_load(f)
        with open(cfg["augmentations"]) as f:
            self.augmentations = yaml.safe_load(f)

        if cfg.get("dataset_name", "omega") == "omega":
            self.datasets = split_datasets(**cfg["datasplitter"])
        elif cfg["dataset_name"] == "libribrain":
            self.datasets = split_datasets_libribrain(**cfg["datasplitter"])
        else:
            raise ValueError(f"Invalid dataset name: {cfg['dataset_name']}")

        # Get model and loss classes dynamically
        model_class = get_model_class(cfg["model_name"])
        loss_class = get_loss_class(cfg["loss_name"])

        self.lit_model = LitModel(
            model_class=model_class,
            loss_class=loss_class,
            model_cfg=model_cfg,
            datasets=self.datasets,
            loss_cfg=cfg["loss"],
            trainer_cfg=cfg["lightning"],
        )

        # Use torchview to visualize the model
        """
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
        """

    def train(self) -> None:
        args = self.dataloader_args
        train_loader = DataLoader(self.datasets.train, shuffle=True, **args)
        val_loader = DataLoader(self.datasets.val, shuffle=False, **args)
        trainer = pl.Trainer(**self.trainer_args)
        trainer.fit(
            self.lit_model, train_loader, val_loader, ckpt_path=self.resume_from
        )

    def test(self) -> None:
        args = self.dataloader_args
        test_loader = DataLoader(self.datasets.test, shuffle=False, **args)
        trainer = pl.Trainer(**self.trainer_args)
        trainer.test(self.lit_model, test_loader)
