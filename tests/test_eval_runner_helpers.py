import numpy as np

from ephys_gpt.training.eval_runner import (
    EvaluationRunner,
    _select_first_chunk_indices,
)
from models.utils import make_dummy_session


def test_select_first_chunk_indices_uses_unique_sessions():
    indices = [
        ("ds", "s1", "0.npy", 0),
        ("ds", "s1", "1.npy", 0),
        ("ds", "s1", "0.npy", 10),
        ("ds", "s2", "0.npy", 0),
        ("ds", "s2", "0.npy", 5),
        ("ds", "s3", "1.npy", 0),
        ("ds", "s3", "0.npy", 0),
    ]

    rng = np.random.default_rng(0)
    selected = _select_first_chunk_indices(indices, num_runs=3, rng=rng)

    assert len(selected) == 3
    sessions = {indices[i][1] for i in selected}
    assert sessions == {"s1", "s2", "s3"}
    assert all(indices[i][2] == "0.npy" for i in selected)
    assert all(indices[i][3] == 0 for i in selected)


def test_build_target_continuation_respects_context(tmp_path):
    runner = object.__new__(EvaluationRunner)

    root = tmp_path / "omega"
    session = "sub-001"
    data = np.arange(2 * 10, dtype=np.float32).reshape(2, 10)
    make_dummy_session(str(root), session, data=data, chunk_idx=0)

    class DummyDataset:
        def __init__(self, base):
            self.root_dirs = {"dataset0": str(base)}
            self.ch_names = ["c0", "c1"]
            self.fill_value = 0

        def _get_session_indices(self, dataset_key, session_name, n_channels):
            return np.arange(n_channels)

    dataset = DummyDataset(root)
    spec = ("dataset0", session, "0.npy", 0)

    context_steps = 2
    steps = 4
    continuation = runner._build_target_continuation(
        dataset, spec, context_steps, steps
    )

    assert continuation is not None
    assert continuation.shape == (2, steps)
    expected = data[:, context_steps : context_steps + steps]
    assert np.array_equal(continuation.numpy(), expected)


def test_max_continuation_across_chunks(tmp_path):
    runner = object.__new__(EvaluationRunner)
    root = tmp_path / "omega"
    session = "sub-002"

    data0 = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)
    data1 = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 100
    make_dummy_session(str(root), session, data=data0, chunk_idx=0)
    make_dummy_session(str(root), session, data=data1, chunk_idx=1)

    class DummyDataset:
        def __init__(self, base):
            self.root_dirs = {"dataset0": str(base)}

    dataset = DummyDataset(root)
    spec = ("dataset0", session, "0.npy", 0)
    context_steps = 2

    max_steps = runner._max_continuation_steps(dataset, spec, context_steps)
    assert max_steps == (6 - context_steps) + 4


def test_build_target_continuation_spans_chunks(tmp_path):
    runner = object.__new__(EvaluationRunner)

    root = tmp_path / "omega"
    session = "sub-003"
    data0 = np.arange(2 * 6, dtype=np.float32).reshape(2, 6)
    data1 = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 100
    make_dummy_session(str(root), session, data=data0, chunk_idx=0)
    make_dummy_session(str(root), session, data=data1, chunk_idx=1)

    class DummyDataset:
        def __init__(self, base):
            self.root_dirs = {"dataset0": str(base)}
            self.ch_names = ["c0", "c1"]
            self.fill_value = 0

        def _get_session_indices(self, dataset_key, session_name, n_channels):
            return np.arange(n_channels)

    dataset = DummyDataset(root)
    spec = ("dataset0", session, "0.npy", 0)
    context_steps = 2
    steps = 8

    continuation = runner._build_target_continuation(
        dataset, spec, context_steps, steps
    )

    assert continuation is not None
    assert continuation.shape == (2, steps)
    expected = np.concatenate(
        [
            data0[:, context_steps:],
            data1[:, : steps - (data0.shape[1] - context_steps)],
        ],
        axis=1,
    )
    assert np.array_equal(continuation.numpy(), expected)


def test_rollout_divergence_zero_when_sequences_match():
    runner = object.__new__(EvaluationRunner)
    gen = np.stack(
        [
            np.sin(np.linspace(0, np.pi, 20, dtype=np.float32)),
            np.cos(np.linspace(0, np.pi, 20, dtype=np.float32)),
        ]
    )
    curves = runner._rollout_divergence_curve(gen, gen, window_steps=5)
    expected = {
        "correlation",
        "covariance",
        "stft_magnitude",
        "stft_angle",
        "fft_magnitude",
        "fft_angle",
    }
    assert expected.issubset(curves.keys())
    for name, curve in curves.items():
        assert curve.size > 0
        assert np.all(np.isfinite(curve))
        assert np.allclose(curve, 0.0, atol=1e-6), name


def test_rollout_divergence_grows_with_noise():
    runner = object.__new__(EvaluationRunner)
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 40, dtype=np.float32)
    target = np.stack([np.sin(t), np.cos(t)])
    noise_scale = np.linspace(0.0, 1.5, t.size, dtype=np.float32)
    noisy = target + rng.standard_normal(target.shape) * noise_scale

    curves = runner._rollout_divergence_curve(noisy, target, window_steps=8)
    corr = curves["correlation"]
    fft_mag = curves["fft_magnitude"]

    assert corr[-1] > corr[0]
    assert corr[-1] > 0.1
    assert fft_mag[-1] > fft_mag[0]
    assert np.all(fft_mag >= 0)


def test_timeseries_divergence_metrics(tmp_path):
    runner = object.__new__(EvaluationRunner)
    runner.sfreq = 10.0

    t = np.linspace(0, 2 * np.pi, 40, dtype=np.float32)
    target = np.stack([np.sin(t), np.cos(t)])
    pred = target * 0.9
    other = np.stack([np.sin(t + 0.5), np.cos(t + 0.5)])

    res = runner._compute_timeseries_divergence_metrics(
        target, pred, other, window_steps=10, out_dir=tmp_path, params={}
    )
    assert res is not None
    assert (tmp_path / "rollout_divergence_timeseries.json").exists()
    assert (tmp_path / "rollout_divergence_timeseries.png").exists()
    assert res["pred"]["psd_jsd"]
    assert "summary" in res


def test_timeseries_divergence_uses_time_axis_when_channels_dominate(tmp_path):
    runner = object.__new__(EvaluationRunner)
    runner.sfreq = 20.0

    timesteps = 10
    channels = 16
    t = np.linspace(0, 1, timesteps, endpoint=False, dtype=np.float32)
    base = np.stack([np.sin(2 * np.pi * (i + 1) * t) for i in range(channels)])
    pred = base * 0.95
    other = np.roll(base, shift=1, axis=1)

    res = runner._compute_timeseries_divergence_metrics(
        base,
        pred,
        other,
        window_steps=5,
        out_dir=tmp_path,
        params={
            "timeseries_divergence": {
                "fs": runner.sfreq,
                "win_seconds": 0.4,
                "stride_seconds": 0.05,
                "fmin": 0.1,
            }
        },
    )

    assert res is not None
    times = np.asarray(res["times_s"], dtype=np.float32)
    win = int(round(0.4 * runner.sfreq))
    stride = max(1, int(round(0.05 * runner.sfreq)))
    expected_times = (
        np.arange(0, timesteps - win + 1, stride, dtype=np.float32) / runner.sfreq
    )
    assert np.allclose(times, expected_times)
    assert len(res["pred"]["psd_jsd"]) == expected_times.size
    assert len(res["pred"]["band_jsd"]) == expected_times.size
    assert len(res["pred"]["stft_wass"]) == expected_times.size


def test_resolve_divergence_window_prefers_seconds_when_available():
    runner = object.__new__(EvaluationRunner)
    runner.sfreq = 100.0

    params_steps = {"divergence_window_steps": 12}
    params_seconds = {"divergence_window_seconds": 0.25}

    assert runner._resolve_divergence_window(params_steps) == 12
    assert runner._resolve_divergence_window(params_seconds) == 25
