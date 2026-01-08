import os
from datetime import datetime, timedelta, timezone

from ephys_gpt.mdl.file_utils import delete_files_before_date


def _set_mtime(path, dt: datetime) -> None:
    ts = dt.timestamp()
    os.utime(path, (ts, ts))


def test_delete_files_before_date(tmp_path) -> None:
    root = tmp_path / "root"
    old_dir = root / "old"
    new_dir = root / "new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()

    old_file = old_dir / "old.txt"
    new_file = new_dir / "new.txt"
    boundary_file = root / "boundary.txt"

    old_file.write_text("old")
    new_file.write_text("new")
    boundary_file.write_text("boundary")

    cutoff = datetime(2024, 1, 15, tzinfo=timezone.utc)
    _set_mtime(old_file, cutoff - timedelta(days=2))
    _set_mtime(new_file, cutoff + timedelta(days=2))
    _set_mtime(boundary_file, cutoff)

    deleted = delete_files_before_date(root, cutoff)

    assert not old_file.exists()
    assert new_file.exists()
    assert boundary_file.exists()
    assert old_dir.is_dir()
    assert new_dir.is_dir()
    assert set(deleted) == {old_file}
