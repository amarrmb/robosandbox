"""`robo-sandbox` CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robo-sandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Scripted Pick — no VLM, no API key")

    run_p = sub.add_parser("run", help="Agent-driven run (stub / openai / ollama / custom)")
    run_p.add_argument("task", help="Natural-language task, e.g. 'pick up the red cube'")
    run_p.add_argument(
        "--vlm-provider",
        choices=["stub", "openai", "ollama", "custom"],
        default="stub",
    )
    run_p.add_argument("--model", default=None)
    run_p.add_argument("--base-url", default=None)
    run_p.add_argument("--api-key-env", default=None)
    run_p.add_argument("--perception", choices=["vlm", "ground_truth"], default=None)
    run_p.add_argument("--max-replans", type=int, default=3)
    run_p.add_argument("--log-level", default="INFO")

    args, rest = p.parse_known_args(argv)
    if args.cmd == "demo":
        from robosandbox.demo import main as demo_main

        return demo_main(rest)
    elif args.cmd == "run":
        from robosandbox.agentic_demo import main as agentic_main

        forwarded = [
            args.task,
            "--vlm-provider",
            args.vlm_provider,
            "--max-replans",
            str(args.max_replans),
            "--log-level",
            args.log_level,
        ]
        if args.model is not None:
            forwarded += ["--model", args.model]
        if args.base_url is not None:
            forwarded += ["--base-url", args.base_url]
        if args.api_key_env is not None:
            forwarded += ["--api-key-env", args.api_key_env]
        if args.perception is not None:
            forwarded += ["--perception", args.perception]
        return agentic_main(forwarded)

    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
