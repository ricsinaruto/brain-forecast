import torch
import torch.nn as nn
import pytest

from ephys_gpt.training.lightning import LitModel, LitModelFreerun


class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DummyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.metrics: dict[str, object] = {}
        self._impl = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        return self._impl(outputs, targets)


def make_litmodel(trainer_cfg: dict) -> LitModel:
    return LitModel(
        model_class=DummyModel,
        loss_class=DummyLoss,
        model_cfg={"input_dim": 4, "output_dim": 3},
        loss_cfg={},
        trainer_cfg=trainer_cfg,
    )


def test_configure_optimizers_builds_scheduler_without_mutation():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "lr_scheduler": {
            "class_name": "ExponentialLR",
            "gamma": 0.9,
            "interval": "epoch",
            "frequency": 2,
            "monitor": "val_loss",
        },
    }
    lit = make_litmodel(trainer_cfg)

    opt_cfg = lit.configure_optimizers()
    scheduler = opt_cfg["lr_scheduler"]["scheduler"]

    assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    # User config should remain intact (class name not popped)
    assert trainer_cfg["lr_scheduler"]["class_name"] == "ExponentialLR"
    assert opt_cfg["lr_scheduler"]["interval"] == "epoch"
    assert opt_cfg["lr_scheduler"]["frequency"] == 2
    assert opt_cfg["lr_scheduler"]["monitor"] == "val_loss"


def test_configure_optimizers_raises_on_unknown_scheduler():
    trainer_cfg = {
        "lr": 1e-3,
        "weight_decay": 0.0,
        "lr_scheduler": {"class_name": "NotAScheduler"},
    }
    lit = make_litmodel(trainer_cfg)

    with pytest.raises(ValueError):
        lit.configure_optimizers()


def test_freerun_config_validation_and_normalisation():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0}
    lit = LitModelFreerun(
        model_class=DummyModel,
        loss_class=DummyLoss,
        model_cfg={"input_dim": 4, "output_dim": 3},
        loss_cfg={},
        trainer_cfg=trainer_cfg,
        free_run_cfg=None,
    )

    cfg = lit._prepare_free_run_cfg(
        {
            "enabled": True,
            "warmup_range": [2, 3],
            "rollout_range": 4,
            "sample_strategy": "sample",
            "temperature": 0.5,
            "log_lengths": True,
        }
    )
    assert cfg["enabled"] is True
    assert cfg["warmup_range"] == (2, 3)
    assert cfg["rollout_range"] == (4, 4)
    assert cfg["sample_strategy"] == "sample"
    assert cfg["temperature"] == 0.5
    assert cfg["log_lengths"] is True

    with pytest.raises(ValueError):
        lit._prepare_free_run_cfg(
            {"enabled": True, "warmup_range": 0, "rollout_range": 1}
        )
    with pytest.raises(ValueError):
        lit._prepare_free_run_cfg(
            {
                "enabled": True,
                "warmup_range": 1,
                "rollout_range": 1,
                "sample_strategy": "greedy",
            }
        )
    with pytest.raises(ValueError):
        lit._prepare_free_run_cfg(
            {"enabled": True, "warmup_range": 1, "rollout_range": 1, "temperature": 0}
        )


def test_test_step_collects_predictions_and_targets():
    trainer_cfg = {"lr": 1e-3, "weight_decay": 0.0}
    lit = make_litmodel(trainer_cfg)
    lit.eval()

    batch = (torch.zeros(2, 4), torch.tensor([0, 1]))
    lit.test_step(batch, batch_idx=0)

    assert len(lit.test_predictions) == 1
    assert len(lit.test_targets) == 1
    assert lit.test_predictions[0].shape[0] == 2
    assert torch.equal(lit.test_targets[0], torch.tensor([0, 1]))
