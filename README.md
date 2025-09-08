# ephys-gpt

Neural signal modeling toolkit for MEG/ephys research. This repo provides:

- Preprocessing pipeline (OSL-ephys stage + custom stage) for Ωmega-style MEG
- Datasets and loaders for discrete, continuous, and image-mapped signals
- A collection of sequence and vision models (GPT-style AR, diffusion, flows, CNN/LSTM, WaveNet, tokenizers, etc.)
- Unified training/eval entrypoints with PyTorch Lightning
- A lightweight test suite that checks shapes and causal gradients


**Install**
- Python: 3.13 (see `setup.py`)
- Create env, then install: `pip install -e .`
- Preprocessing requires `osl-ephys` (install separately) and MNE. See `requirements.txt` for other deps.


**Repo Structure**
- `ephys_gpt/preprocessing`: OSL-ephys wrapper + custom stage-2 transforms (normalize, clip, µ-law)
- `ephys_gpt/dataset`: Datasets, augmentations, split utilities, mixup dataloader
- `ephys_gpt/models`: Model zoo (GPT2MEG, ST-GPT, BENDR, MEGFormer, NTD, Wavenets, VQ/VAE tokenizers, etc.)
- `ephys_gpt/layers`: Attention blocks, embeddings, quantizers and model submodules
- `ephys_gpt/losses` and `ephys_gpt/metrics`: Losses (CE, MSE, diffusion, VQ) and classification metrics
- `ephys_gpt/training`: Lightning `LitModel`, experiment runners, dynamic model/loss resolution
- `ephys_gpt/eval`: Evaluation tasks for quantized AR, diffusion, flow/image and BENDR models
- `ephys_gpt/utils`: Quantizers, plotting (PSD/cov), sampling utils, test helpers
- Entrypoints: `preprocess.py` (preproc) and `run.py` (train/test/eval/tokenizer)


**Data Preprocessing**
Stage 1 runs the OSL-ephys pipeline, Stage 2 applies custom transforms and saves chunked `.npy` arrays.

- Config template: `configs/local/preprocess.yaml`
- Example command:
  - `python preprocess.py --dataset omega --stage both --args configs/local/preprocess.yaml`
- Key options in the YAML:
  - `data_path`: raw dataset root (Ωmega layout)
  - `osl_config`: OSL-ephys YAML (optional; defaults provided in code)
  - `n_workers`: Dask workers for OSL
  - `chunk_seconds`: window length for saved chunks
  - `preproc_config`: stage-2 steps, e.g. `normalize`, `clip`, `mulaw_quantize`
- Outputs: `preprocessed/<sub-...>/<i>.npy` each holding `data`, `sfreq`, `ch_names`, `pos_2d`, etc.

Notes
- Requires `osl-ephys` for Stage 1: `from osl_ephys.preprocessing import run_proc_batch`.
- `Omega` loader discovers `sub-*/ses-*/meg/*.ds` folders and builds subject/session names.


**Training**
Use `run.py` with a train YAML. Trainer/dataloader/model/loss are configured from file.

- Example (Ωmega + GPT2MEG):
  - `python run.py --mode train --args configs/gpt2meg/train.yaml`
- Example (LibriBrain + CNNLSTM):
  - `python run.py --mode train --args configs/cnnlstm/train.yaml`
- Resume/testing:
  - `python run.py --mode test --args <same yaml>`

Train YAML (common keys)
- `model_config`: path to model-specific config (see `configs/**/model.yaml`)
- `augmentations`: optional YAML for dataset augs
- `save_dir`, `resume_from`
- `model_name`, `loss_name` (resolved via `ephys_gpt/training/utils.py`)
- `dataset_name`: `omega` (default) or `libribrain`
- `datasplitter`: dataset factory and parameters (see below)
- `dataloader`: `batch_size`, `num_workers`, `pin_memory`, etc.
- `loss`: hyperparams (e.g., `label_smoothing` for CE)
- `lightning`: optimizer/lr/compile/logging options
- `trainer`: PyTorch Lightning `Trainer` args (accelerator, precision, epochs)


**Evaluation**
Evaluations are run via the same entrypoint with `--mode eval` and an eval YAML.

- Example (quantized AR):
  - `python run.py --mode eval --args configs/gpt2meg/eval.yaml`
- Select eval runner in YAML with `eval_class` (default `EvalQuant`). Implemented classes:
  - `EvalQuant`: quantized AR (GPT2MEG/STGPT2MEG/VQ-GPT). Computes history-sweep and N-step curves, free-run PSD/cov.
  - `EvalDiffusion`: continuous diffusion models (NTD): horizon MSE, history sweep, and recursive forecasting.
  - `EvalFlow`: image-space AR (MEGFormer): maps back to channels for MSE curves and free-run.
  - `EvalBENDR`: continuous 1-step/N-step metrics for BENDRForecast.
  - `EvalVQ`: tokeniser + forecaster systems (reconstructs predicted latent codes).
- Eval YAML extras (`eval` block): `ckpt_path` (required), `future_steps`, `gen_seconds`, sampling (`gen_sampling`, `top_p`, `temperature`), `accelerator`, and dataset shape hints (`channel_shape`, `num_channels`).
- Outputs: saved under `<save_dir>/evals` (curves as `.npy` and `.pdf/.png`, PSD/cov for generated/test sequences).


**Datasets**
Constructed via `ephys_gpt.dataset.datasplitter`. Two families:

- Chunked Ωmega examples:
  - `ChunkDataset`: quantized discrete targets; returns `(inputs[:, :-1], targets[:, 1:])` per channel
  - `ChunkDatasetForecastCont`: continuous forecasting pairs as float tensors
  - `ChunkDatasetReconstruction`: returns `(x, pos_2d, ch_type)` → `x` (for models requiring sensor metadata)
  - `ChunkDatasetImage`: maps sensors to sparse `H×W×T` image via 2‑D positions; returns `(img, img)` for reconstruction/forecasting
  - `ChunkDatasetImageQuantized`: image version with integer tokens and one‑step shift
  - Helper: `build_indices` scans `preprocessed/` sessions and builds `(session, chunk_idx, start)` triplets
  - `split_datasets(...)` splits sessions into train/val/test and instantiates the chosen dataset class

- LibriBrain grouped classification:
  - `split_datasets_libribrain(...)` wraps `pnpl.datasets.LibriBrainPhoneme`
  - `RandomLabelGroupedDataset`: samples a label then k examples with that label on the fly
  - `GroupedDatasetAugmented`: same with optional augmentations and µ‑law tokenization

Loaders
- Standard `torch.utils.data.DataLoader` or `MixupDataLoader` (mixup with optional quantization on-the-fly)


**Models (brief)**
- Autoregressive (quantized): `GPT2MEG`, `STGPT2MEG`, `VQGPT2MEG` (tokenizer + AR transformer)
- Tokenizers/VQ: `VideoGPTTokenizer`, `Emu3VisionVQ`, `BrainTokenizer` (RVQ-like)
- Continuous forecasting: `BENDRForecast` (encoder-decoder), `NTD` (diffusion)
- Image/flow AR: `MEGFormer` (image patches + flows)
- CNN/LSTM hybrids: `CNNLSTM` (+ optional attention pooling, transformer blocks)
- Causal convs: `WavenetFullChannel`, `Wavenet3D`
- 3D/vision stacks: `LITRA`, `TACA`, `CK3D`, `TASA3D`, `ChronoFlowSSM`, `LatteAR`

Model configs live under `configs/**/model.yaml` and are fed to constructors by `ExperimentDL`.


**Losses & Metrics**
- Losses: `CrossEntropy`, `MSE`, `NLL` (flows), `VQVAELoss`, `VQNSPLoss`, `BrainTokenizerLoss`
- Metrics (classification): accuracy, top‑k accuracy, micro‑F1


**CLI Entry Points**
- Preprocess: `python preprocess.py --dataset omega --stage <stage_1|stage_2|both> --args <yaml>`
- Train: `python run.py --mode train --args <yaml>`
- Test: `python run.py --mode test --args <yaml>`
- Eval: `python run.py --mode eval --args <yaml>` (optionally set `eval_class` in YAML)
- Tokenizer fitting (FAST example): set a config for `ExperimentTokenizer` and run `python run.py --mode tokenizer --args <yaml>`


**Configurations**
- Examples: `configs/gpt2meg/*.yaml`, `configs/stgpt2meg/*.yaml`, `configs/cnnlstm/*.yaml`, `configs/models/*.yaml`, `configs/local/*.yaml`
- For LibriBrain, see `configs/cnnlstm/train.yaml` (`dataset_name: libribrain` + `GroupedDatasetAugmented`)
- For Ωmega AR, point `datasplitter.dataset_root` at `preprocessed/` from the preproc step


**Tests**
- Run: `pytest -q`
- What’s covered:
  - Dataset builders: shapes, shifting semantics, image mapping
  - Model forwards for GPT2MEG/STGPT2MEG/MEGFormer/BENDR/NTD/WaveNet/tokenizers
  - Causality/gradient tests: `tests/utils.assert_future_grad_zero` verifies future‑proof gradients across models
  - Lightweight synthetic inputs; CPU by default


**Contributing**
- Style: `flake8` (`.flake8` config), keep changes minimal and focused
- Add tests for new models/datasets; mirror existing shape/causality checks
- Plug‑in points when adding features:
  - Models: implement under `ephys_gpt/models/`, export in `models/__init__.py`, and map in `training/utils.py:get_model_class`
  - Losses: add under `ephys_gpt/losses/` and map in `training/utils.py:get_loss_class`
  - Datasets: add class under `ephys_gpt/dataset/` and register in `dataset/datasplitter.py:DATASET_CLASSES`
  - Eval: implement under `ephys_gpt/eval/` and, if invoked via CLI, map name → class in `run.py`
- Open a PR with a concise description, config snippet, and a passing `pytest`


**Extras**
- `scripts/` includes dataset download helpers and Modal cloud orchestration (`modal_app.py`). These are optional utilities.

Known notes
- Some example configs may reference paths you’ll need to adjust (e.g., model config path under `configs/models/`).
- Certain eval runners are marked “not tested” in code comments; prefer `EvalQuant` unless your model matches the specialized interface.


**Model Details**
- GPT2MEG: Channel‑wise next‑token GPT built on HuggingFace GPT‑2 blocks; expects quantized integer tokens `x ∈ {0…V−1}` shaped `(B, C, T)` and returns logits `(B, C, T, V)` for predicting `x[..., t+1]` from past. Supports `n_positions` context windows and can run with cached KV states during generation; embeddings include per‑channel and optional conditioning.
- STGPT2MEG: Spatio‑temporal transformer with channel mixing inside blocks; input/output are the same as GPT2MEG `(B, C, T) → (B, C, T, V)` but attention operates jointly over time and channels for better cross‑sensor coupling. Use for quantized AR on Ωmega chunks with richer cross‑channel dynamics.
- VQGPT2MEG: Two‑stage pipeline that tokenizes a sequence of sparse topomap images and then autoregresses over latent codes; expects topomap tensors `(B, H, W, T)` (or tokens if `train_tokenizer=False`) and outputs `(logits, targets)` for next‑token prediction. Works with an internal VQ tokenizer (e.g., Emu3VisionVQ) or a checkpoint loaded via `tokenizer_path`.
- BrainOmniSystem (Tokenizer + Forecast): First converts raw MEG `(x, pos, ch_type)` into residual VQ tokens with `BrainOmniTokenizer`, then predicts next tokens with `BrainOmniForecast`; input is `(B, C, T)` with sensor metadata and output is `(logits, codes_tgt)` shaped `(B, C_latent, Nq, T, K)`. Use when you want a discrete latent representation with a stage‑wise token predictor; tokenizer can be frozen or co‑trained.
- MEGFormer: Image‑space AR with normalizing flows over patch tokens; expects image sequences `(B, H, W, T)` produced by `ChunkDatasetImage` and returns `(nll_per_token, logdet)` during training. `forecast(ctx, steps)` autoregressively samples future frames, which are mapped back to channels using dataset pixel indices for evaluation.
- BENDRForecast: Continuous 1‑step forecaster inspired by BENDR; accepts `(x, pos, ch_type)` where `x ∈ ℝ^{B×C×T}` and returns predicted next‑sample sequences `(B, C, T_enc_or_raw)` used with MSE loss. Provides `forecast(past, horizon)` to grow sequences autoregressively using the learned 1‑step predictor; receptive field is set by the conv downsampling stack.
- NTD (diffusion): Continuous diffusion over multi‑channel signals; forward returns `(noise, pred_noise, mask)` for NLL/MSE‑like training, and `forecast(past, horizon)` generates future samples in `(B, C, Lp+N)`. Inputs are real‑valued `(B, C, L)`, optional conditioning channels and forecast masks are supported via `mask_channel/p_forecast`.
- WavenetFullChannel: Causal dilated 1‑D convs over quantized channels; input is integer tokens `(B, C, T)` and output is logits `(B, C, T, Q)` where Q is the number of quantization levels. Optional global/local conditioning and channel‑wise learned token embeddings are supported.
- Wavenet3D: Spatio‑temporal WaveNet operating on `(B, H, W, T)` integer tokens with 3‑D dilated causal convolutions over time and 2‑D kernels over space; outputs `(B, H, W, T, Q)` logits. Use with `ChunkDatasetImageQuantized` for topomap‑token AR.
- CNNLSTM: Hybrid conv frontend + optional BiLSTM + transformer blocks with attention pooling or “last” pooling; expects continuous `(B, C, T)` and returns classification logits `(B, num_classes)`. Designed for label classification tasks (e.g., LibriBrain phonemes) with optional mixup in the dataloader.
- Baseline CNNs: `CNNMultivariate`/`CNNUnivariate` apply causal 1‑D conv stacks to continuous inputs `(B, C, T)` and output same‑shape reconstructions/forecasts; `*Quantized` variants embed integer tokens and output `(B, C, T, V)` logits. Useful as light baselines for sanity checks and ablations.
- VideoGPTTokenizer: VQ‑VAE for `(T, H, W)` sequences; accepts `(H, W, T)` or `(B, H, W, T)` and returns reconstruction plus VQ outputs (`encodings`, `embeddings`). Used to turn topomap videos into discrete token grids; decode returns `(B, H, W, T)`.
- Emu3VisionVQ: Alternative video tokenizer (configurable embed/channels/codebook size) suitable for `(H, W, T)` inputs from `ChunkDatasetImage`; encode returns latent quantized codes and embeddings, decode reconstructs to `(B, H, W, T)`. Pair with VQ‑AR forecasters such as VQGPT2MEG.
- BrainOmniTokenizer: Residual‑vector‑quantizer for raw MEG with multiple codebook stages; input is `(x, pos, ch_type)` and output includes latent tokens `(B, C_latent, T, Nq)` and optional reconstructions. Intended to be coupled with `BrainOmniForecast` but usable standalone for representation learning.
- ChronoFlowSSM: Flow‑based sequence model on videos `[B, T, C, H, W]∈[0,1]` with spatial flows, a temporal state‑space backbone, and a conditional emission flow; trains by exact NLL in logit space and supports long‑horizon sampling via `sample(x_context, steps)`. Useful for learned simulators and video‑like brain maps.
- LITRA / TACA / CK3D / TASA3D: 3‑D/planar attention stacks for quantized signals; typically accept integer token grids `(B, H, W, T)` along with optional dense embeddings and return `(B, H, W, T, Q)` logits. These are research models for spatio‑temporal AR with different factorisations (tri‑plane attention, coarse‑to‑fine 3‑D, etc.).
