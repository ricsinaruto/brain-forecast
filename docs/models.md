# Models and tokenizers

This repository bundles autoregressive transformers, convolutional forecasters, diffusion/flow models, classifiers, and multiple tokenizer families. This document summarizes each model/tokenizer file, its architecture, and input/output interfaces.

## Autoregressive transformers (`gpt2meg.py`)
- **GPT2MEG**: Wraps HuggingFace `GPT2Model` with custom `Embeddings` that combine quantized tokens, channel IDs, optional conditioning, and subject IDs. Input `x` is `(B, C, T)` quantized codes; output logits reshape to `(B, C, T, V)` where `V` is `vocab_size`/`quant_levels`. Supports cached `past_key_values` for incremental decoding.
- **GPT2MEG_Trf**: Pure PyTorch transformer built from `TransformerBlock` stacks. Accepts `(B, C, T)` token IDs or precomputed embeddings and returns logits reshaped to `(B, C, T, V)`. Optional causal flag controls attention masking.
- **GPT2MEG_Cond**: Adds conditional tokens and global embeddings (e.g., subject or trial descriptors) via `TransformerBlockCond`. Inputs include `n_cond_tok` tokens `(B, n_cond_tok, T)` and optional global embeddings; outputs logits `(B, C, T, V)`.
- **GPT2MEGMix / STGPT2MEG / VQGPT2MEG**: Variants that swap transformer block types (spatiotemporal blocks, mixture-of-experts) or integrate vector-quantized tokenizers for residual VQ decoding. Interfaces mirror `(B, C, T)` quantized inputs -> `(B, C, T, V)` logits.

## Convolutional autoregressive models
- **WavenetFullChannel / Wavenet3D (`wavenet.py`)**: Stacks causal dilated convolutions with gated activations, residual and skip paths. Inputs `(B, Q, T)` quantized channels (1D) or `(B, Q, T, H, W)` for 3D; outputs logits `(B, Q, T)` (or `(B, Q, T, H, W)` flattened) through `WavenetLogitsHead`/`Wavenet3DLogitsHead`.
- **CNNLSTM (`cnnlstm.py`)**: Combines per-channel convolutions, pooling, and LSTM heads for classification or autoregression. Accepts `(B, C, T)` signals; classifier head outputs class logits `(B, num_classes)` while autoregressive variants emit `(B, C, T, V)`.
- **BENDRForecast (`bendr.py`)**: Encoder-decoder with causal convolutions and contextual transformer layers. Inputs `(B, C, T)` continuous signals; decoder predicts future quantized steps `(B, C, T, V)` using causal transpose convolutions.
- **CK3D (`ck3d.py`)**: Causal 3D convolutional kernels with attention-style mixing and up/downsampling pyramids for video-like MEG tokens. Inputs `(B, C, T, H, W)`; outputs either logits `(B, C, T, V)` via mixture heads or reconstructed volumes for autoregression.
- **FlatGPT (`flatgpt.py`)**: Flattens video tokens or residual VQ codes and feeds them to transformer blocks (can wrap HuggingFace video adapters). Input tokens `(B, T, D)` or `(B, C, T)` quantized maps; outputs logits `(B, T, V)` reshaped back to channel format when needed.

## Diffusion and flow models
- **NTD (`ntd.py`)**: Noise-to-data diffusion model using masked 1D convolutions, channel and timestep embeddings, and adaptive convolution blocks. Inputs continuous `(B, C, T)` signals with noise levels; outputs predicted noise or denoised signals `(B, C, T)` for diffusion training objectives.
- **MEGFormer / JetViTFlow (`megformer.py`)**: Flow-based architecture that patches signals (`PatchSignal`), passes them through ViT-style coupling layers (`JetNVP`, `ViTCoupling`), and optionally emits Gaussian mixture parameters via `GMMHead`. Inputs `(B, C, T)` continuous signals; outputs transformed tensors for flow log-likelihoods or sampled reconstructions with matching shape.
- **ChronoFlowSSM (`chronoflow.py`)**: Hierarchical normalizing flow combining actnorm, invertible 1x1 convs, affine couplings, squeezing, and selective SSM temporal backbones. Inputs `(B, C, T, H, W)` spatiotemporal tensors; forward returns latent variables/log-determinants for likelihood estimation and inverse reconstructs data with the same shape.

## Attention stacks and research variants
- **LITRA (`litra.py`)**: Axis-aware positional embeddings with alternating spatial/temporal attention blocks and optional memory-based variants (`LiTrALayerMem`). Inputs `(B, C, T, H, W)` video/MEG tensors; outputs logits or embeddings `(B, T, D)` before task-specific heads.
- **TACA (`taca.py`)**: Axial attention encoder with memory compression and decoder for video-like sequences. Inputs `(B, C, T, H, W)`; outputs logits `(B, C, T, V)` or decoded frames.
- **TASA3D (`tasa3d.py`)**: Lightweight 3D attention stack used as a building block for other models (e.g., FlatGPT video adapters). Input/outputs follow `(B, C, T, H, W)` feature maps.
- **BrainOmniCausalForecast (`brainomni.py`)**: Aligns BrainOmni tokenizer outputs with causal cross-attention forecasters; inputs `(B, C, T, D)` token embeddings and returns logits `(B, C, T, V)`.
- **NSR (`nsr.py`)**: State-space recurrent model with OscSSM cells and RMSNorm. Inputs `(B, C, T)` continuous signals; outputs autoregressive predictions `(B, C, T, V)` or continuous reconstructions.

## Classification wrappers
- **`classifier.py`**: Wraps encoders for continuous or quantized inputs. `ClassifierContinuous` consumes `(B, C, T)` float signals and outputs class logits `(B, num_classes)`; `ClassifierQuantized` and `ClassifierQuantizedImage` accept token IDs or image-shaped codes and pool over time before classification.
- **`baselines.py`**: CNN baselines (`CNNMultivariate`, `CNNUnivariate` and quantized variants) that take `(B, C, T)` and emit `(B, num_classes)` logits.

## Tokenizers (`models/tokenizers/`)
- **Emu3VisionVQ (`emu3.py`)**: Causal 3D VQ-VAE for MEG/video. Encoder uses causal 3D convs with temporal/spatial norms; decoder mirrors with causal transposed convs. Vector quantizer supports residual VQ (`Emu3RVQ`). Inputs `(B, C, T, H, W)` floats; encoder outputs discrete codes `(B, T', H', W')` and decoder reconstructs continuous tensors matching the input shape.
- **BrainOmniCausalTokenizer (`brainomni.py`)**: Causal encoder/decoder pair that maps continuous `(B, C, T)` signals to autoregressive token sequences (`CausalTokenSequence`) and back. Outputs token IDs `(B, T)` (or `(B, C, T)` when channelized) with logits for codebook entries; decoder reconstructs waveforms.
- **Factorized MEG autoencoder (`factorized.py`)**: Patch-based encoder/decoder using temporal down/upsampling blocks and HuggingFace `PretrainedConfig`. Inputs `(B, C, T)`; encoder produces latent codes and quantized outputs `(B, T_latent, code_dim)`, decoder reconstructs `(B, C, T)`.
- **Flat tokenizers (`flat_tokenizers.py`)**: Simple amplitude/BPE/block-causal tokenizers operating on `(B, C, T)` sequences to produce discrete token IDs `(B, T_tokens)` and inverse transforms back to waveforms when applicable.
- **Cosmos tokenizer (`cosmos.py`)**: Vision-tokenizer-style patching/quantization for spatiotemporal inputs `(B, C, T, H, W)` yielding token grids `(B, T', H', W')` with codebook logits.
- **MODULAR/NeuroRVQ reference (`modularvq.py`, `neurorvq_reference/`)**: Reference residual VQ stacks with patching/unpatching helpers for causal video tokens. Inputs `(B, C, T, H, W)`; outputs residual codebooks (`n_levels` code tensors) and reconstructed videos.
- **MOVQGAN (`movqgan/`)**: GAN-based residual vector quantizer with LPIPS perceptual loss. Encodes `(B, C, T, H, W)` videos to codebooks and decodes reconstructions; discriminator heads operate on the same shapes.
- **HF adapters (`hf_adapters/*.py`)**: Thin wrappers for HuggingFace LLM/VLM configs (e.g., `Qwen2_5_Video`, `Gemma3`) exposing a unified forward signature for token or video inputs to integrate with `FlatGPT` and downstream trainers.

## Input/output conventions
- Time-major tensors are generally `(B, C, T)` for channel-first signals and `(B, C, T, H, W)` for spatiotemporal data.
- Quantized models output logits over vocab/codebook dimensions, typically reshaped to align with input channels or spatial grids (e.g., `(B, C, T, V)` or `(B, T', H', W', V)`).
- Continuous models (diffusion/flow/autoencoders) return tensors matching the input shape alongside auxiliary losses/log-determinants as needed.
- Conditional models accept optional conditioning tensors (e.g., `cond` in GPT2MEG, global embeddings in `TransformerBlockCond`) that broadcast over time or channels.
