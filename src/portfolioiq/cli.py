"""Command line interface: ``python -m portfolioiq.cli <command> --config configs/sample.yaml``."""

from __future__ import annotations

import argparse
import sys
import time

from portfolioiq.config import load_config
from portfolioiq.pipeline import STEPS, run_step
from portfolioiq.utils import get_logger, setup_logging, write_json

log = get_logger("portfolioiq.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="portfolioiq")
    parser.add_argument("command", choices=[s.replace("_", "-") for s in STEPS] + ["run-all"])
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--from-step", default=None, help="run-all: start at this step")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(log_file=cfg.path("logs_dir") / f"pipeline_{cfg.name}_{stamp}.jsonl")
    steps = list(STEPS) if args.command == "run-all" else [args.command.replace("-", "_")]
    if args.from_step:
        steps = steps[steps.index(args.from_step.replace("-", "_")):]
    timings = {}
    for s in steps:
        timings[s] = round(run_step(s, cfg), 1)
    if args.command == "run-all":
        write_json({"config": cfg.name, "timings_seconds": timings, "total_seconds": round(sum(timings.values()), 1)},
                   cfg.reports_dir / f"timings_{cfg.name}.json")
    log.info("done", steps=steps, timings=timings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
