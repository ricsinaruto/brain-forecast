# Modal commands

`run.py` exposes multiple modes that share the same YAML-driven configuration system. Examples below assume configs live under `configs/`.

- **Train**: `python run.py --mode train --args <train_yaml>`
  - Merges top-level trainer settings with nested `model_config` and `datasplitter` blocks.
  - Supports Lightning options such as `max_steps`, `accumulate_grad_batches`, AMP, and `compile`.
- **Test**: `python run.py --mode test --args <train_yaml>`
  - Rebuilds the validation/test dataloader from the training config and evaluates the restored checkpoint passed via `--ckpt_path` or config `resume_from`.
- **Eval**: `python run.py --mode eval --args <eval_yaml>`
  - Dispatches to `ephys_gpt.eval` classes (`EvalQuant`, `EvalDiffusion`, `EvalFlow`, `EvalCont`, `EvalVQ`, `EvalFlat`, `EvalText`).
  - `eval` block controls checkpoint selection (`ckpt_path`, `version`, `step`, `best`) and generation settings (`temperature`, `top_k`, `top_p`, `future_steps`, `gen_seconds`).
- **Tokenizer (vision/MEG)**: `python run.py --mode tokenizer --args <tokenizer_yaml>`
  - Trains tokenizers such as `Emu3VisionVQ`, `BrainOmniCausalTokenizer`, or factorized MEG autoencoders.
  - Uses `tokenizer_data` and `text_dataloader` sections to point at image/video shards or MEG chunks.
- **Tokenizer (text)**: `python run.py --mode tokenizer-text --args <text_tokenizer_yaml>`
  - Handles BPE or delimiter-based text vocab training with the same experiment scaffolding.

Use `--dict --args '<json>'` to supply inline configurations for quick experiments.
