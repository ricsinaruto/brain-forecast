# ephys-gpt

Neural signal modeling toolkit for MEG/ephys research. It bundles preprocessing, dataset utilities, tokenizers, autoregressive/diffusion/flow/conv models, and PyTorch Lightning training + evaluation entrypoints.

## Install
- Python 3.13 (see `setup.py`).
- Create/activate an environment, then `pip install -e .`.
- Preprocessing depends on `osl-ephys` and MNE in addition to the packages in `requirements.txt`.

## Repository layout
- `preprocess.py` – CLI for multi-stage MEG preprocessing.
- `run.py` – Unified train/test/eval/tokenizer driver built on `ExperimentDL`.
- `evals.py` – Lightweight evaluation runner for automated checkpoint sweeps.
- `ephys_gpt/preprocessing` – Dataset-specific preprocessors (Ωmega/MEG and MOUS) with OSL wrappers and post-processing steps.
- `ephys_gpt/dataset` – Dataset builders, datasplitters, augmentations, mixup dataloader, and text helpers.
- `ephys_gpt/models` – Model zoo (GPT-style AR, Wavenets, CNN/LSTM, diffusion/flow, tokenizers, MEGFormer, BrainOmni, etc.) plus `layers/` and `mdl/` submodules.
- `ephys_gpt/training` – Lightning `LitModel`, experiment runners, optimizer/scheduler/pl logging glue, tokenizer trainers, and evaluation utilities.
- `ephys_gpt/eval` – Evaluation classes invoked via `run.py --mode eval` (quantized AR, diffusion, flow/image, continuous, flat, text, VQ systems).
- `ephys_gpt/utils` – Quantizers, plotting, sampling, metrics, and test helpers.
- `tests/` – Shape/causality checks for datasets and models.
- `configs/` – Example configs for preprocessing, models, training/eval, augmentations, and local development.

## Running the main pipelines
### Preprocessing (`preprocess.py`)
The preprocessing script orchestrates OSL-ephys (stage 1) and custom transforms (stage 2/3) for different datasets.

```bash
python preprocess.py --dataset <omega|mous|mous_conditioned> \
  --stage <stage_1|stage_2|stage_3|both|all> \
  --args configs/local/preprocess.yaml
```
- `--dataset` selects the dataset-specific pipeline (Ωmega, MOUS, or MOUS with conditioning).
- `--stage` chooses which stages to run (`both` = stage_1+stage_2, `all` runs all available stages).
- `--args` points to a YAML file containing stage parameters.

Outputs are stored under `preprocessed/<sub-...>/<i>.npy` with data arrays and metadata (`sfreq`, `ch_names`, `pos_2d`, etc.).

#### Preprocess config anatomy (`configs/local/preprocess.yaml`)
- **Paths**: `data_path` (raw dataset root), `out_root` (optional override for `preprocessed/`), `osl_config` (OSL YAML path or inline dict to forward to `osl_ephys.preprocessing.run_proc_chain`).
- **Stage selection**: `stage_1` keys mirror OSL-ephys parameters (filtering, ICA, bad channel detection). `stage_2` handles chunking/normalization/quantization via `preproc_config`. `stage_3` handles post-processing such as conditioning labels for MOUS.
- **Performance**: `n_workers` (Dask workers), `max_bad_channels` and `max_chunks` to bound work, `chunk_seconds` to control sequence length for downstream models.
- **Transforms**: `preproc_config` toggles `normalize`, `clip`, `mulaw_quantize`, `random_sign_flip`, etc. Use `mulaw_quantize` parameters to set codebook size for downstream quantizers.
- **Metadata**: `save_info` enables saving sensor metadata alongside `.npy` chunks for plotting/evaluation.

### Training / testing / evaluation (`run.py`)
`run.py` loads a YAML config, resolves model/dataset/loss classes, and dispatches to the requested mode.

```bash
python run.py --mode train --args configs/gpt2meg/train.yaml
python run.py --mode test  --args configs/gpt2meg/train.yaml
python run.py --mode eval  --args configs/gpt2meg/eval.yaml
python run.py --mode tokenizer --args <tokenizer_yaml>
python run.py --mode tokenizer-text --args <text_tokenizer_yaml>
```

#### Training/test config anatomy
- **Composition**: top-level YAML (`configs/*/train.yaml`) references a `model_config` (e.g., `configs/gpt2meg/model.yaml`) that defines architecture + tokenizer settings. Fields in the top-level file override/extend the nested model file when merged.
- **Datasets**: `datasplitter` chooses a dataset class (`OmegaDataset`, `QuantizedOmegaDataset`, `MousDataset`, etc.) plus paths and sizing (`dataset_root`, `example_seconds`, `step_seconds`, `val_ratio`, `example_overlap_seconds`). `dataset_kwargs` configures quantization bins, positional encodings, and conditioning labels.
- **Dataloaders**: `dataloader` configures batch size, workers, and persistence. For tokenizer training, use `text_dataloader` and `tokenizer_data` blocks to specify text corpora or image shards.
- **Models/losses**: `model_name` and `loss_name` are strings resolved by `training/utils.py`. `loss` holds task-specific knobs (e.g., `label_smoothing`, `alpha_l1`, KL weights). `model_config` adds architecture depth/width, attention/windowing, quantizer heads, and tokenizer checkpoints.
- **Optim/Lightning**: `lightning` carries optimizer choice (`optimizer`, `lr`, `weight_decay`, betas), scheduler (`lr_scheduler`, `warmup_steps`), AMP/compile toggles, gradient clipping, and logging frequency.
- **Trainer**: `trainer` mirrors PyTorch Lightning arguments (accelerator, devices, precision, `max_epochs`, `max_steps`, `accumulate_grad_batches`, checkpointing/logging cadence).
- **Saving/resume**: `save_dir`, `resume_from`, and `ckpt_path` control where checkpoints and logs land. Use `version` to segregate runs under the same save directory.

#### Evaluation config anatomy (`run.py --mode eval`)
- `eval_class` chooses the evaluator (`EvalQuant`, `EvalDiffusion`, `EvalFlow`, `EvalCont`, `EvalVQ`, `EvalFlat`, `EvalText`).
- `eval` block configures checkpoint selection (`ckpt_path`, `version`, `step`, `best`), data sizing (`max_batches`, `num_examples`), and generation settings (`gen_sampling`, `temperature`, `top_k`, `top_p`, `future_steps`, `gen_seconds`).
- `plot`/`sample` options enable PSD plots, spectrograms, and sample dumping. `shape_hints` can enforce target sequence length or channels for diffusion/flow/image evaluators.

### Automated evaluation runner (`evals.py`)
`evals.py` is a thin wrapper around `ephys_gpt.training.eval_runner.EvaluationRunner`, useful for periodic checkpoint checks.

```bash
# YAML config
python evals.py --args configs/gpt2meg/eval.yaml

# Inline JSON config
python evals.py --dict --args '{"save_dir": "...", "eval_runner": {"ckpt_path": "..."}, ...}'
```

`eval_runner` options support selecting a checkpoint (`ckpt_path`), version/step filtering, `max_batches`, `num_examples`, and optional generation settings under `eval_runner.generate` (strategy, temperature, top-k/p). The runner rebuilds the validation dataloader from the provided `datasplitter`/`dataloader` configuration before scoring.

### Configuration tips
- Start from the closest example under `configs/` (e.g., `configs/gpt2meg/train.yaml` for Ωmega AR training, `configs/cnnlstm/train.yaml` for LibriBrain classification, `configs/local/preprocess.yaml` for preprocessing). Copy and adjust paths rather than editing the originals.
- Keep architecture in `model_config` and data/training knobs in the top-level file. Shared defaults can be placed in a base YAML and extended via YAML anchors/`include` if desired.
- Quantized pipelines: set `datasplitter.dataset_root` to `preprocessed/` outputs and align tokenizer vocab/codebook sizes between `preproc_config`, `dataset_kwargs`, and the model config.
- Tokenizer training: use `mode tokenizer`/`tokenizer-text` with configs under `configs/tokenizers/`. These mirror the train configs but swap in `ExperimentTokenizer`/`ExperimentTokenizerText` and text/image dataset specs.
- Remote runs: adjust `trainer.accelerator`/`devices` for multi-GPU, and ensure `save_dir` points to a shared filesystem; Lightning loggers are wired in `training/logging`.

## Module quick reference
- **Preprocessing**: dataset wrappers (`Omega`, `MOUS`, `MOUSConditioned`) orchestrating OSL-ephys and custom transforms.
- **Datasets**: chunked Ωmega loaders (continuous/quantized/image), LibriBrain grouped classification helpers, text loaders, and a `MixupDataLoader`.
- **Models**: GPT2MEG/STGPT2MEG/VQGPT2MEG, Wavenet variants, CNN/LSTM hybrids, diffusion (`NTD`), flow/image models (`MEGFormer`, `ChronoFlowSSM`), tokenizers (`VideoGPTTokenizer`, `Emu3VisionVQ`, `BrainOmniTokenizer`), and research attention stacks (LITRA/TACA/CK3D/TASA3D/LatteAR).
- **Layers/MDL**: reusable attention/convolution blocks plus model wrappers in `ephys_gpt.layers` and `ephys_gpt.mdl` (e.g., `Perceiver`, `LinearAttention`, `S4`, `CKConv`, `MLPMixer`).
- **Losses**: task-specific objectives in `ephys_gpt.losses` (e.g., contrastive, reconstruction, classification, multi-task losses).
- **Training**: `ExperimentDL`, `ExperimentTokenizer`, and `ExperimentTokenizerText` wire configs into Lightning modules, optimizers, schedulers, and loggers; `eval_runner` and evaluation classes provide reusable scoring/generation hooks.
- **Eval**: task-specific evaluators (`EvalQuant`, `EvalDiffusion`, `EvalFlow`, `EvalCont`, `EvalVQ`, `EvalFlat`, `EvalText`) that share dataset splitting logic and plotting/sampling utilities.
- **Logging**: wrappers for WandB/CSV/TensorBoard in `ephys_gpt.logging` plus experiment helpers under `scripts/`.
- **Utils**: quantizers (µ-law, residual VQ helpers), PSD/cov plotting, sampling helpers, YAML loaders, and gradient-causality assertions for tests.
- **Notebooks/Scripts**: exploratory notebooks and utility scripts (e.g., dataset inspection, checkpoint export) under `notebooks/` and `scripts/`.

## Tests
Run the lightweight test suite with:

```bash
pytest -q
```

The suite checks dataset shapes/shift semantics/image mapping, forward passes across key models (GPT2MEG, STGPT2MEG, MEGFormer, BENDR, NTD, Wavenet, tokenizers, etc.), and causality/gradient safety via utilities like `tests/utils.assert_future_grad_zero`. Tests use synthetic CPU inputs by default.

## Contributing
- Open issues or discussions before large changes; small fixes/docs are welcome via PR.
- Keep README/config examples synchronized with new pipelines or arguments.
- Ensure new models/datasets include minimal tests and synthetic-forward coverage. Prefer CPU-friendly fixtures in `tests/`.
- Run `pytest -q` (and tokenizer-specific smoke tests if you add a tokenizer) before submitting.
