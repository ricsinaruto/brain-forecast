# Architecture Overview

This project organises MEG/ephys preprocessing, datasets, tokenizers, models, and training utilities into cohesive packages. The sections below outline the primary modules, notable classes, and how components fit together.

## Top-level entrypoints
- **`preprocess.py`**: CLI orchestrating dataset-specific preprocessing pipelines (Ωmega, MOUS) that invoke OSL-ephys for stage-1 cleaning followed by chunking/normalisation and optional conditioning steps.
- **`run.py`**: Unified driver for training, testing, evaluation, and tokenizer training. It merges YAML configs, resolves model/loss/dataset factories, and dispatches to PyTorch Lightning experiment classes.
- **`evals.py`**: Thin wrapper around `training.eval_runner.EvaluationRunner` to sweep checkpoints with the same dataset/dataloader config logic used by `run.py`.

## `ephys_gpt.preprocessing`
Dataset-specific preprocessing stages. Pipelines load raw recordings, apply OSL-ephys routines, chunk sequences, normalise or quantize signals, and optionally add conditioning labels. Outputs include `.npy` tensors plus metadata (sampling frequency, channel names, 2D sensor positions) used by downstream datasets.

## `ephys_gpt.dataset`
- **`datasplitter.py`**: Builds train/val/test splits and canonical channel layouts from preprocessed roots. Handles cross-session mapping and optional conditioning labels.
- **`datasets.py`**: Core datasets. `ChunkDataset` loads chunked `.npy` arrays with augmentations, encodes channel types, and reshapes image-style inputs back to channel-major tensors via `Postprocessor`. Exposes `__getitem__` returning `(tensor, target)` pairs with shape `(C, T)` or `(H, W, T)` depending on the dataset.
- **`augmentations.py`**: Lightweight amplitude scaling, random sign flip, jitter, and dropout transforms configured from YAML.
- **`dataloaders.py`**: Mixup-enabled dataloaders and helper collates for quantized/continuous targets. Supports optional channel masking and padding.
- **`libribrain.py`**: Grouped classification helpers for LibriBrain-style datasets, mapping audio transcripts to MEG snippets and supporting subject-level splits.

## `ephys_gpt.layers` and `ephys_gpt.mdl`
Reusable building blocks shared across models. Includes transformer attention variants, spatiotemporal blocks (e.g., `STGPTBlock`, `TransformerBlockCond`), convolutional layers, S4/SSM primitives, Perceiver blocks, and projection heads. These modules define the core computation units used in GPT-style, diffusion, and flow models.

## `ephys_gpt.models`
Model zoo covering autoregressive transformers, Wavenet-style convolutions, diffusion/flow models, classification heads, and tokenizers. Key files include:
- **`gpt2meg.py`**: GPT-2 based quantized forecasters (`GPT2MEG`, `GPT2MEG_Trf`, `GPT2MEG_Cond`, `STGPT2MEG`, `VQGPT2MEG`) with channel-aware embeddings and optional conditioning tokens.
- **`wavenet.py`**: 1D and 3D causal convolution stacks (`WavenetFullChannel`, `Wavenet3D`) with gated residual blocks and logits heads producing `(B, Q, T)` predictions.
- **`cnnllstm.py`**: CNN + LSTM hybrids for classification or autoregressive logits, including 3D Wavenet-inspired classifiers.
- **`bendr.py`**: BENDR-style contextual encoder (`ConvEncoderBENDR`) and autoregressive forecaster `BENDRForecast` with causal transposed decoding.
- **`ntd.py`**: Diffusion model (`NTD`) built from masked convolutions, timestep/channel embedders, and adaptive convolution blocks for continuous reconstruction.
- **`megformer.py`**: Flow-based MEGFormer (`JetViTFlow`/`MEGFormer`) combining ViT-style patching with coupling layers and optional GMM output heads.
- **`chronoflow.py`**: Normalizing flows for spatiotemporal data (`ChronoFlowSSM`) with actnorm, invertible convolutions, coupling layers, and selective SSM temporal backbones.
- **`ck3d.py`**: Causal 3D convolutional kernels with pyramid up/downsampling and mixture heads for video-like autoregression.
- **`flatgpt.py`**: Flat transformer front-ends for video or residual VQ tokens (`FlatGPT`, `FlatGPTRVQ`, embedding-only variants) that can wrap HuggingFace video models.
- **`litra.py` / `taca.py` / `tasa3d.py`**: Research spatiotemporal attention stacks and axial attention memory compressors for long-context modeling.
- **`brainomni.py`**: `BrainOmniCausalForecast` forecaster aligning Omni tokenizers with causal cross-attention.
- **`baselines.py`**: Lightweight CNN baselines (univariate/multivariate, quantized/continuous variants) for quick comparison.
- **`classifier.py`**: Classification wrappers that attach logits heads to pretrained quantized or continuous encoders.
- **Tokenizers (`models/tokenizers/`)**: Vision and MEG tokenizers such as `Emu3VisionVQ`, `BrainOmniCausalTokenizer`, factorized MEG autoencoders, µ-law/block causal tokenizers, and reference RVQ implementations. See `models.md` for details.

## `ephys_gpt.training`
- **`lightning.py`**: `LitModel`/`LitModelFreerun` Lightning modules that wrap a model and loss class, handle free-run rollouts, metrics logging, and optimizer/scheduler construction.
- **`experiment.py` & experiment helpers**: Factories that resolve model/loss/dataset classes from config strings, build dataloaders, and instantiate Lightning trainers. Includes tokenizer-specific experiments.
- **`eval_runner.py`**: Rebuilds dataloaders from config and runs evaluator classes against checkpoints, supporting generation sampling options.

## `ephys_gpt.eval`
Evaluator classes invoked via `run.py --mode eval`: quantized AR (`EvalQuant`), diffusion (`EvalDiffusion`), flow/image (`EvalFlow`), continuous (`EvalCont`), residual VQ (`EvalVQ`), flattened token models (`EvalFlat`), and text models (`EvalText`). They share plotting, PSD, and sampling hooks and reuse dataset construction from training configs.

## `ephys_gpt.losses`
Collection of task-specific objectives (classification, reconstruction, contrastive, diffusion/flow losses). Each loss exposes metrics used by `LitModel` for logging.

## `ephys_gpt.utils`
Utilities for quantization (`mulaw_torch`, residual VQ helpers), plotting (PSD, covariance), sampling (autoregressive decoding utilities), YAML loading, logging wrappers, and gradient-causality asserts used in tests.

## Logging and scripts
- **`ephys_gpt.logging`**: WandB/CSV/TensorBoard loggers and helpers for experiment metadata.
- **`scripts/`**: One-off utilities (dataset inspection, checkpoint export, debug helpers) that consume the same configs as the main entrypoints.
