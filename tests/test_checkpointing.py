from pathlib import Path
from types import SimpleNamespace

from ephys_gpt.training.checkpointing import ThreadedModelCheckpoint


def test_threaded_checkpoint_respects_epoch_cadence(tmp_path):
    saved = []

    def after_save(path, *_args):
        saved.append(Path(path).name)

    class DummyTrainer:
        def __init__(self, epoch):
            self.current_epoch = epoch
            self.global_step = 0
            self.is_global_zero = True
            self.logger = SimpleNamespace(version=0)
            self.loggers = []
            self.lightning_module = SimpleNamespace()

        def save_checkpoint(self, filepath, save_weights_only=False):
            Path(filepath).write_text("ckpt")

    cb = ThreadedModelCheckpoint(
        dirpath=tmp_path,
        filename="ckpt",
        save_top_k=-1,
        epoch_cadence=2,
        after_save=after_save,
    )

    # Epoch 1 (index 0) should be skipped due to cadence
    trainer = DummyTrainer(epoch=0)
    cb._save_checkpoint(trainer, str(tmp_path / "ckpt.ckpt"))
    cb.on_train_end(trainer, None)
    assert not saved
    assert not list(tmp_path.glob("*.ckpt"))

    # Epoch 2 triggers save with formatted filename
    trainer = DummyTrainer(epoch=1)
    cb._save_checkpoint(trainer, str(tmp_path / "ckpt.ckpt"))
    cb.on_train_end(trainer, None)

    files = list(tmp_path.glob("ckpt-epoch00002.ckpt"))
    assert files, "checkpoint should be written on cadence"
    assert saved == ["ckpt-epoch00002.ckpt"]
