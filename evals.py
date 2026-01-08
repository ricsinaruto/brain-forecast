import argparse
import yaml
import json
import torch
import mne

mne.set_log_level("WARNING")
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from ephys_gpt.training.eval_runner import EvaluationRunner  # noqa: E402


def main(cli_args=None):
    # parse arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--args", type=str, help="args file name", required=True)
    parser.add_argument(
        "-d", "--dict", action="store_true", help="dict mode", default=False
    )

    script_args = parser.parse_args(cli_args)
    args_file = script_args.args

    if script_args.dict:
        # conver json string to dict using json.loads
        cfg = json.loads(args_file)
    else:
        with open(args_file) as f:
            cfg = yaml.safe_load(f)

    EvaluationRunner(cfg).run()


if __name__ == "__main__":
    main()
