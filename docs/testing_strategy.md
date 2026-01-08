# Testing strategy

The repository ships with a lightweight CPU-friendly test suite under `tests/` focused on shape/causality and smoke coverage for datasets, tokenizers, and models.

## Goals
- Validate dataset splitting, augmentation, and channel alignment logic with synthetic `.npy` chunks.
- Exercise forward passes of key models (GPT2MEG/STGPT2MEG/VQGPT2MEG, MEGFormer, BENDR, NTD, Wavenet, tokenizers) to ensure tensor shapes and causal masking are respected.
- Provide gradient-causality checks via `ephys_gpt.utils.tests.assert_future_grad_zero` to guard against information leakage.

## How to run
- Activate the project environment (see `requirements.txt`) and run `pytest -q` from the repo root.
- Tests write temporary data to the local filesystem and do not require GPUs.
- Additional tokenizer-specific configs can be exercised manually by pointing `run.py --mode tokenizer` at the configs in `configs/tokenizers/` if you add new tokenizers.

## What to watch for
- Causal attention/convolution layers should zero gradients for future timesteps (see GPT2MEG and Wavenet tests).
- Dataset tests expect small, synthetic shapes; when adding new datasets ensure fixtures keep runtime low.
- Flow/diffusion tests rely on CPU defaults; avoid adding GPU-only code paths without guards or skips.
