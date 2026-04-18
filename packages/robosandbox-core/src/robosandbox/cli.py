"""`robo-sandbox` CLI entry point.

Dispatches to one of the top-level demos. A full YAML-driven `run`
subcommand lands in v0.2; for v0.1 we just expose the two built-in
demos that prove the plumbing end-to-end.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robo-sandbox", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Scripted Pick — no VLM, no API key")
    run_p = sub.add_parser("run", help="VLM-driven agent")
    run_p.add_argument("task", help="Natural-language task, e.g. 'pick up the red cube'")
    run_p.add_argument("--model", default="gpt-4o-mini")
    run_p.add_argument("--base-url", default=None)
    run_p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    run_p.add_argument("--perception", choices=["vlm", "ground_truth"], default="vlm")
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
            "--model",
            args.model,
            "--api-key-env",
            args.api_key_env,
            "--perception",
            args.perception,
            "--max-replans",
            str(args.max_replans),
            "--log-level",
            args.log_level,
        ]
        if args.base_url is not None:
            forwarded += ["--base-url", args.base_url]
        return agentic_main(forwarded)

    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
