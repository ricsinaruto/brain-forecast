import argparse
import yaml

import torch
import mne

mne.set_log_level("WARNING")
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from ephys_gpt import (  # noqa: E402
    ExperimentTokenizer,
    ExperimentDL,
    ExperimentTokenizerText,
    ExperimentIBQ,
    ExperimentVidtok,
    EvalQuant,
    EvalDiffusion,
    EvalFlow,
    EvalCont,
    EvalVQ,
    EvalFlat,
    EvalText,
)


eval_class_map = {
    "EvalQuant": EvalQuant,
    "EvalDiffusion": EvalDiffusion,
    "EvalFlow": EvalFlow,
    "EvalCont": EvalCont,
    "EvalVQ": EvalVQ,
    "EvalFlat": EvalFlat,
    "EvalText": EvalText,
}


def main(cli_args=None):
    # parse arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", type=str, help="args file name", required=True)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="run mode",
        required=True,
        choices=[
            "train",
            "test",
            "eval",
            "tokenizer",
            "tokenizer-text",
            "ibq",
            "vidtok",
        ],
    )

    script_args = parser.parse_args(cli_args)
    args_file = script_args.args
    mode = script_args.mode

    with open(args_file) as f:
        cfg = yaml.safe_load(f)

    if mode == "train":
        ExperimentDL(cfg).train()
    elif mode == "test":
        ExperimentDL(cfg).test()
    elif mode == "eval":
        eval_key = cfg.get("eval_class", "EvalQuant")
        eval_class = eval_class_map.get(eval_key, EvalQuant)
        eval_class(cfg).run_all()
    elif mode == "tokenizer":
        ExperimentTokenizer(cfg)
    elif mode == "tokenizer-text":
        ExperimentTokenizerText(cfg)
    elif mode == "ibq":
        ExperimentIBQ(cfg).train()
    elif mode == "vidtok":
        ExperimentVidtok(cfg).train()
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
