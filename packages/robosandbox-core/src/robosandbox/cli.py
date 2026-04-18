"""`robo-sandbox` CLI entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robo-sandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Scripted Pick — no VLM, no API key")

    run_p = sub.add_parser("run", help="Agent-driven run (stub / openai / ollama / custom)")
    run_p.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Natural-language task, e.g. 'pick up the red cube' (planner path)",
    )
    run_p.add_argument(
        "--task",
        dest="task_flag",
        default=None,
        help="Built-in task name (required with --policy, e.g. 'home', 'pick_cube')",
    )
    run_p.add_argument(
        "--policy",
        default=None,
        help="Path to a policy checkpoint or recorded-episode directory. "
             "When set, bypasses the planner and drives the sim open-loop.",
    )
    run_p.add_argument("--max-steps", type=int, default=1000)
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
        if args.policy is not None:
            return _run_policy_cli(args)

        if args.task is None:
            run_p.error("task is required unless --policy is set")
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


def _run_policy_cli(args: argparse.Namespace) -> int:
    """Policy-in-the-loop branch. Bypasses the planner entirely."""
    import logging
    import time
    from pathlib import Path

    from robosandbox.policy import load_policy, run_policy
    from robosandbox.sim.mujoco_backend import MuJoCoBackend
    from robosandbox.tasks.loader import load_builtin_task

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    task_name = args.task_flag or args.task
    if task_name is None:
        print("[run --policy] --task <builtin_task> is required", file=sys.stderr)
        return 2

    try:
        task = load_builtin_task(task_name)
    except FileNotFoundError as e:
        print(f"[run --policy] {e}", file=sys.stderr)
        return 2

    try:
        policy = load_policy(Path(args.policy))
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"[run --policy] failed to load policy: {e}", file=sys.stderr)
        return 2

    sim = MuJoCoBackend(render_size=(240, 320), camera="scene")
    sim.load(task.scene)
    try:
        # Settle gravity so the initial observation is at rest.
        for _ in range(100):
            sim.step()
        t0 = time.time()
        result = run_policy(
            sim, policy, max_steps=args.max_steps, success=task.success
        )
        elapsed = time.time() - t0
    finally:
        sim.close()

    success_ok = result["success"]
    verdict = "success" if success_ok else ("failure" if success_ok is False else "unknown")
    final_reason = (
        f"policy_completed_{result['steps']}_steps"
        if result["steps"] >= args.max_steps
        else "policy_exhausted_trajectory"
    )
    print(f"[run --policy] task:        {task.name}")
    print(f"[run --policy] policy:      {args.policy}")
    print(f"[run --policy] verdict:     {verdict}")
    print(f"[run --policy] steps:       {result['steps']}")
    print(f"[run --policy] final_reason: {final_reason}")
    print(f"[run --policy] wall:        {elapsed:.1f}s")
    return 0 if success_ok else 1


if __name__ == "__main__":
    sys.exit(main())
