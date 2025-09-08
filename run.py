import argparse
import yaml

import torch
import mne

mne.set_log_level("WARNING")
torch.set_float32_matmul_precision("medium")

from ephys_gpt import (  # noqa: E402
    ExperimentTokenizer,
    ExperimentDL,
    EvalQuant,
    EvalDiffusion,
    EvalFlow,
    EvalBENDR,
    EvalVQ,
)


eval_class_map = {
    "EvalQuant": EvalQuant,
    "EvalDiffusion": EvalDiffusion,
    "EvalFlow": EvalFlow,
    "EvalBENDR": EvalBENDR,
    "EvalVQ": EvalVQ,
}


def main(cli_args=None):
    # parse arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", type=str, default="", help="args file name")
    parser.add_argument(
        "-m", "--mode", type=str, default="", help="run mode", required=True
    )

    script_args = parser.parse_args(cli_args)
    args_file = script_args.args
    mode = script_args.mode

    config_path = args_file or "configs/train.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if mode == "train":
        ExperimentDL(cfg).train()
    elif mode == "test":
        ExperimentDL(cfg).test()
    elif mode == "eval":
        # choose correct class based on config, default to Evals
        eval_key = cfg.get("eval_class", "EvalQuant")
        eval_class = eval_class_map.get(eval_key, EvalQuant)
        eval_class(cfg).run_all()
    elif mode == "tokenizer":
        ExperimentTokenizer(cfg)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
