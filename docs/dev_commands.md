# Developer commands

Common developer workflows use the Python entrypoints and YAML configs in `configs/`.

- **Install in editable mode**: `pip install -e .` (Python 3.13; see `requirements.txt` for dependencies). Consider installing `osl-ephys` and `mne` for preprocessing.
- **Run preprocessing**: `python preprocess.py --dataset <omega|mous|mous_conditioned> --stage <stage_1|stage_2|stage_3|both|all> --args configs/local/preprocess.yaml`
  - Produces chunked `.npy` files under `preprocessed/` with optional metadata (channel names, sampling freq, 2D positions).
- **Train/Eval model**: `python run.py --mode train --args configs/gpt2meg/train.yaml` (replace with desired config). Use `--mode test` for checkpoint evaluation and `--mode eval` for evaluator-driven sweeps.
- **Automated checkpoint sweeps**: `python evals.py --args configs/gpt2meg/eval.yaml` to run periodic validation or sampling.
- **Tokenizer training**: `python run.py --mode tokenizer --args configs/tokenizers/<tokenizer>.yaml` for MEG/video tokenizers or `--mode tokenizer-text` for text.
- **Export/analysis scripts**: scripts under `scripts/` (e.g., dataset inspection, checkpoint export) can be run directly with `python scripts/<name>.py` using the same config conventions.
- **Lint/tests**: lightweight test suite via `pytest -q`; no repo-wide linter configuration is provided, but formatters (e.g., `black`) can be run manually if desired.
