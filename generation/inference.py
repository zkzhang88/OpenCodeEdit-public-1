#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__:
    from .inference_core import create_run, resume_run, show_status
else:
    from inference_core import create_run, resume_run, show_status


EXECUTORS = ("api", "siliconflow-batch", "llm-infer")


def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-completion", type=int)
    parser.add_argument("--max-batch-attempts", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--visible-devices")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--thinking", dest="thinking", action="store_true")
    thinking.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.set_defaults(thinking=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable multi-round inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Create and start a new inference run.")
    run.add_argument("--executor", required=True, choices=EXECUTORS)
    run.add_argument("--model", required=True, help="Model profile name from the config.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--wait", action="store_true")
    _add_runtime_overrides(run)

    resume = subparsers.add_parser("resume", help="Advance an existing run.")
    resume.add_argument("--run-dir", type=Path, required=True)
    resume.add_argument("--wait", action="store_true")
    resume.add_argument("--retry-failed", action="store_true")

    status = subparsers.add_parser("status", help="Show persisted run status.")
    status.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            override_names = (
                "num_completion",
                "max_batch_attempts",
                "temperature",
                "top_p",
                "max_tokens",
                "visible_devices",
                "concurrency",
                "tensor_parallel_size",
                "thinking",
            )
            overrides = {name: getattr(args, name) for name in override_names}
            return create_run(
                executor=args.executor,
                model=args.model,
                config_path=args.config,
                input_path=args.input,
                output_path=args.output,
                run_dir=args.run_dir,
                wait=args.wait,
                overrides=overrides,
            )
        if args.command == "resume":
            return resume_run(
                args.run_dir, wait=args.wait, retry_failed=args.retry_failed
            )
        print(json.dumps(show_status(args.run_dir), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
