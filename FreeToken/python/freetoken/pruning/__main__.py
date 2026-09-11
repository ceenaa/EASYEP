"""Run with python -m freetoken.pruning --help."""

import argparse

from .artifacts import all_ones_mask, checkpoint_identity, make_mask, read_json, write_new
from .workflow import compare, evaluate, prepare, replay, reviews
from .primevul import import_primevul


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description="Experimental EASYEP-style V4 expert masking for defensive code review")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("import-primevul", help="Import an aligned source ZIP into the test split; labels remain unknown")
    p.add_argument("--archive", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(run=import_primevul)
    p = commands.add_parser("all-ones", help="Create an identity-bound full mask for parity testing")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=lambda a: write_new(a.output, all_ones_mask(checkpoint_identity(a.model_dir))))
    p = commands.add_parser("prepare", help="Read source cases and render checkpoint-native prompts")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--split", choices=["calibration", "validation", "test"], required=True)
    p.add_argument("--max-bytes", type=positive, default=100000)
    p.add_argument("--max-prompt-tokens", type=positive, default=4096)
    p.add_argument("--thinking-mode", choices=["thinking", "chat"], default="thinking")
    p.add_argument("--output", required=True)
    p.set_defaults(run=prepare)
    for name, function in (("reviews", reviews), ("replay", replay)):
        p = commands.add_parser(name)
        p.add_argument("--input", required=True)
        p.add_argument("--server", default="http://127.0.0.1:1919")
        p.add_argument("--output", required=True)
        p.add_argument("--timeout", type=positive, default=3600)
        if name == "reviews":
            p.add_argument("--max-tokens", type=positive, default=2048)
        else:
            p.add_argument("--model-dir", required=True)
            p.add_argument("--max-seq-len", type=positive, default=8192)
        p.set_defaults(run=function)
    p = commands.add_parser("mask", help="Rank experts after verifying complete replay coverage")
    p.add_argument("--stats", required=True)
    p.add_argument("--replay-log", required=True)
    p.add_argument("--keep", type=positive, required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=lambda a: write_new(a.output, make_mask(read_json(a.stats), read_json(a.replay_log), a.keep)))
    p = commands.add_parser("compare", help="Prepare paired held-out reviews for human grading")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=compare)
    p = commands.add_parser("evaluate", help="Compute metrics from completed human judgments")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
