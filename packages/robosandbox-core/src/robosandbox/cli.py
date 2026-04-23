"""`robo-sandbox` CLI entry point."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="robo-sandbox")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("demo", help="Scripted Pick — no VLM, no API key")

    viewer_p = sub.add_parser(
        "viewer", help="Browser live viewer (requires `pip install 'robosandbox[viewer]'`)"
    )
    viewer_p.add_argument("--host", default="127.0.0.1")
    viewer_p.add_argument("--port", type=int, default=8000)
    viewer_p.add_argument(
        "--task",
        default="pick_cube_franka",
        help="Built-in task to preload on startup",
    )
    viewer_p.add_argument(
        "--runs-dir",
        default="runs",
        help="Directory to write recorded episodes (default: ./runs)",
    )
    viewer_p.add_argument("--sim-backend", default="mujoco", choices=["mujoco", "newton"])
    viewer_p.add_argument("--viser-port", type=int, default=8090,
                          help="Port for Newton's Viser 3D viewer (Newton mode only)")
    viewer_p.add_argument("--device", default="cuda:0",
                          help="Compute device for Newton backend (Newton mode only)")

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
    run_p.add_argument("--sim-backend", default="mujoco")
    run_p.add_argument("--sim-viewer", default="null")
    run_p.add_argument("--viewer-port", type=int, default=8080)
    run_p.add_argument("--device", default="cuda:0")
    run_p.add_argument("--log-level", default="INFO")

    sim_p = sub.add_parser(
        "simulate",
        help="Load a built-in task in a sim backend and step it forward",
    )
    sim_p.add_argument("--task", default="pick_cube_franka")
    sim_p.add_argument("--sim-backend", default="mujoco")
    sim_p.add_argument("--steps", type=int, default=240)
    sim_p.add_argument("--render-height", type=int, default=480)
    sim_p.add_argument("--render-width", type=int, default=640)
    sim_p.add_argument("--sim-viewer", default="null")
    sim_p.add_argument("--viewer-port", type=int, default=8080)
    sim_p.add_argument("--device", default="cuda:0")
    sim_p.add_argument("--hold", action="store_true")

    exp_p = sub.add_parser(
        "export-lerobot",
        help="Convert a recorded episode directory to LeRobot v3 dataset format",
    )
    exp_p.add_argument("src", help="Source episode dir (e.g. runs/20260101-120000-abcd1234)")
    exp_p.add_argument("dst", help="Destination LeRobot dataset dir")
    exp_p.add_argument("--task", default=None, help="Override task string")
    exp_p.add_argument("--fps", type=int, default=30)

    train_p = sub.add_parser(
        "train",
        help="Train a PPO policy from scratch via Newton GPU-parallel RL (no demonstrations needed)",
    )
    train_p.add_argument("--task", required=True, help="Built-in task name (e.g. pick_cube_franka)")
    train_p.add_argument("--world-count", type=int, default=512, help="Parallel Newton worlds")
    train_p.add_argument("--total-steps", type=int, default=5_000_000, help="Total env steps to train")
    train_p.add_argument("--n-steps", type=int, default=256, help="Steps per rollout per world")
    train_p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_p.add_argument("--delta-scale", type=float, default=0.05,
                         help="Max joint angle change per step (radians)")
    train_p.add_argument("--settle-steps", type=int, default=50,
                         help="Physics settle steps after each reset")
    train_p.add_argument("--device", default="cuda:0", help="GPU device for Newton + PyTorch")
    train_p.add_argument("--output", default="runs/rl/policy", help="Output checkpoint directory")
    train_p.add_argument("--log-interval", type=int, default=5, help="Log every N iterations")

    eval_p = sub.add_parser(
        "eval",
        help="GPU-parallel policy evaluation via Newton multi-world backend",
    )
    eval_p.add_argument("--task", required=True, help="Built-in task name (e.g. pick_cube_franka)")
    eval_p.add_argument("--policy", required=True, help="Path to policy checkpoint or episode dir")
    eval_p.add_argument("--sim-backend", default="newton", choices=["mujoco", "newton"],
                        help="mujoco = single world + RGB; newton = N parallel worlds, state only")
    eval_p.add_argument("--world-count", type=int, default=64,
                        help="Parallel worlds (newton only, ignored for mujoco)")
    eval_p.add_argument("--max-steps", type=int, default=500, help="Max sim steps per world")
    eval_p.add_argument("--settle-steps", type=int, default=100, help="Physics settle steps before eval")
    eval_p.add_argument("--device", default="cuda:0", help="GPU device for Newton")

    viz_p = sub.add_parser(
        "download-franka-visuals",
        help="Download full-resolution Franka visual meshes from mujoco_menagerie "
             "(~33 MB) to ~/.cache/robosandbox/franka_visuals/",
    )
    viz_p.add_argument(
        "--cache-dir",
        default=None,
        help="Override cache root (default: $ROBOSANDBOX_CACHE/franka_visuals "
             "or ~/.cache/robosandbox/franka_visuals)",
    )
    viz_p.add_argument("--force", action="store_true", help="Re-download even if cached")

    args, rest = p.parse_known_args(argv)
    if args.cmd == "demo":
        from robosandbox.demo import main as demo_main

        return demo_main(rest)
    elif args.cmd == "viewer":
        from robosandbox.viewer.server import run as viewer_run

        backend = args.sim_backend
        print(f"RoboSandbox viewer [{backend}] — open http://{args.host}:{args.port}")
        if backend == "newton":
            print(f"Newton Viser 3D viewer — http://127.0.0.1:{args.viser_port}")
        viewer_run(
            host=args.host, port=args.port, initial_task=args.task,
            runs_dir=args.runs_dir, sim_backend=backend, viser_port=args.viser_port,
            device=getattr(args, "device", "cuda:0"),
        )
        return 0
    elif args.cmd == "run":
        if args.policy is not None:
            return _run_policy_cli(args)

        if args.task is None:
            run_p.error("task is required unless --policy is set")
        if args.sim_backend != "mujoco":
            run_p.error(
                "planner-driven runs currently require --sim-backend mujoco; "
                "use `simulate` or `run --policy` for Newton"
            )
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
    elif args.cmd == "simulate":
        return _simulate_cli(args)
    elif args.cmd == "export-lerobot":
        from pathlib import Path

        from robosandbox.export.lerobot import export_episode

        out = export_episode(
            Path(args.src),
            Path(args.dst),
            task=args.task,
            fps=args.fps,
        )
        print(f"Exported LeRobot dataset to: {out}")
        out_abs = Path(out).resolve()
        ckpt_dir = out_abs.parent / f"{out_abs.name}-checkpoint"
        print()
        print("Next steps:")
        print()
        print("  # 1. Train an ACT policy on this dataset:")
        print(
            f"  python -m lerobot.scripts.train \\\n"
            f"      --dataset.repo_id=local \\\n"
            f"      --dataset.root={out_abs} \\\n"
            f"      --policy.type=act \\\n"
            f"      --output_dir={ckpt_dir}"
        )
        print()
        print("  # 2. Evaluate the trained checkpoint:")
        print(
            f"  robo-sandbox eval \\\n"
            f"      --task {args.task or '<task_name>'} \\\n"
            f"      --policy {ckpt_dir} \\\n"
            f"      --sim-backend mujoco"
        )
        print()
        print(
            "  Note: Newton backend is state-only (no camera), so vision "
            "policies (ACT, Diffusion) need --sim-backend mujoco."
        )
        return 0
    elif args.cmd == "train":
        return _train_ppo_cli(args)
    elif args.cmd == "eval":
        return _eval_parallel_cli(args)
    elif args.cmd == "download-franka-visuals":
        from robosandbox.assets.franka_visuals import cli as download_cli

        return download_cli(cache_dir=args.cache_dir, force=args.force)

    p.print_help()
    return 2


def _run_policy_cli(args: argparse.Namespace) -> int:
    """Policy-in-the-loop branch. Bypasses the planner entirely."""
    import logging
    import time
    from pathlib import Path

    from robosandbox.policy import load_policy, run_policy
    from robosandbox.sim import create_sim_backend
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

    sim = create_sim_backend(
        args.sim_backend,
        render_size=(240, 320),
        camera="scene",
        viewer=args.sim_viewer,
        port=args.viewer_port,
        device=args.device,
    )
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
        f"policy_step_limit_reached_{result['steps']}"
        if result["steps"] >= args.max_steps
        else f"policy_trajectory_done_{result['steps']}"
    )
    print(f"[run --policy] task:        {task.name}")
    print(f"[run --policy] policy:      {args.policy}")
    print(f"[run --policy] sim_backend: {args.sim_backend}")
    print(f"[run --policy] verdict:     {verdict}")
    print(f"[run --policy] steps:       {result['steps']}")
    print(f"[run --policy] final_reason: {final_reason}")
    print(f"[run --policy] wall:        {elapsed:.1f}s")
    return 0 if success_ok else 1


def _simulate_cli(args: argparse.Namespace) -> int:
    import time

    from robosandbox.sim import create_sim_backend
    from robosandbox.tasks.loader import load_builtin_task

    try:
        task = load_builtin_task(args.task)
    except FileNotFoundError as e:
        print(f"[simulate] {e}", file=sys.stderr)
        return 2

    try:
        sim = create_sim_backend(
            args.sim_backend,
            render_size=(args.render_height, args.render_width),
            camera="scene",
            viewer=args.sim_viewer,
            port=args.viewer_port,
            device=args.device,
        )
    except (ImportError, ValueError) as e:
        print(f"[simulate] failed to create backend: {e}", file=sys.stderr)
        return 2

    sim.load(task.scene)
    try:
        for _ in range(args.steps):
            sim.step()
        obs = sim.observe()
        print(f"[simulate] task:        {task.name}")
        print(f"[simulate] sim_backend: {args.sim_backend}")
        print(f"[simulate] steps:       {args.steps}")
        print(f"[simulate] joints:      {np.array2string(obs.robot_joints, precision=4)}")
        for oid, pose in obs.scene_objects.items():
            print(
                "[simulate] object:      "
                f"{oid} xyz=({pose.xyz[0]:.4f}, {pose.xyz[1]:.4f}, {pose.xyz[2]:.4f})"
            )
        if args.sim_backend == "newton" and args.sim_viewer == "viser":
            print(f"[simulate] viewer_url:  http://127.0.0.1:{args.viewer_port}")
            if args.hold:
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    pass
    finally:
        sim.close()
    return 0


def _train_ppo_cli(args: argparse.Namespace) -> int:
    """RL training: PPO from scratch via Newton GPU-parallel worlds."""
    from pathlib import Path

    from robosandbox.sim import create_sim_backend
    from robosandbox.tasks.loader import load_builtin_task

    try:
        task = load_builtin_task(args.task)
    except FileNotFoundError as e:
        print(f"[train] {e}", file=sys.stderr)
        return 2

    if "newton" not in task.supported_backends:
        print(
            f"[train] task {args.task!r} does not support newton "
            f"(supported: {', '.join(task.supported_backends)})",
            file=sys.stderr,
        )
        return 2

    if task.success.data.get("kind") is None:
        print(
            f"[train] task {args.task!r} has no success criterion — cannot compute reward",
            file=sys.stderr,
        )
        return 2

    try:
        from robosandbox.rl.ppo import train_ppo
    except ImportError as e:
        print(
            f"[train] RL training requires PyTorch.\n"
            f"  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            f"  ({e})",
            file=sys.stderr,
        )
        return 2

    print(f"[train] task:          {task.name}")
    print(f"[train] world_count:   {args.world_count}")
    print(f"[train] total_steps:   {args.total_steps:,}")
    print(f"[train] output:        {args.output}")

    try:
        sim = create_sim_backend(
            "newton",
            render_size=(240, 320),
            camera="scene",
            viewer="null",
            device=args.device,
            world_count=args.world_count,
        )
    except (ImportError, ValueError) as e:
        print(f"[train] failed to create Newton backend: {e}", file=sys.stderr)
        return 2

    sim.load(task.scene)
    try:
        train_ppo(
            sim,
            task,
            total_steps=args.total_steps,
            n_steps=args.n_steps,
            lr=args.lr,
            delta_scale=args.delta_scale,
            settle_steps=args.settle_steps,
            device=args.device,
            save_path=Path(args.output),
            log_interval=args.log_interval,
        )
    finally:
        sim.close()

    return 0


def _policy_image_inputs(policy) -> set[str]:
    """Return the set of visual input feature keys a policy expects, if any.

    LeRobot policies expose this via ``policy.config.input_features``. The
    :class:`~robosandbox.policy.LeRobotPolicyAdapter` wraps the real policy
    on its private ``_policy`` attribute, so we unwrap when needed.

    Detection prefers the feature spec's ``.type == FeatureType.VISUAL``
    marker (works for both the legacy ``observation.image`` key and the
    namespaced ``observation.images.<cam>`` family). Falls back to a
    name-prefix match when the type field is absent. Returns an empty
    set for any non-LeRobot policy — no false-positive warnings.
    """
    inner = getattr(policy, "_policy", policy)
    cfg = getattr(inner, "config", None)
    feats = getattr(cfg, "input_features", None) if cfg is not None else None
    if not feats:
        return set()
    try:
        items = list(feats.items())
    except AttributeError:
        return set()
    visual = set()
    for key, spec in items:
        if not isinstance(key, str):
            continue
        # Prefer the typed marker when present.
        ftype = getattr(spec, "type", None)
        type_name = getattr(ftype, "name", None) or getattr(ftype, "value", None) or str(ftype)
        if isinstance(type_name, str) and type_name.upper() == "VISUAL":
            visual.add(key)
            continue
        # Fallback for shapes without a typed feature spec.
        if key.startswith("observation.image"):
            visual.add(key)
    return visual


def _eval_parallel_cli(args: argparse.Namespace) -> int:
    """Evaluate a policy in sim — MuJoCo (single world) or Newton (N parallel worlds)."""
    from pathlib import Path

    from robosandbox.policy import load_policy, run_eval_parallel, run_policy
    from robosandbox.sim import create_sim_backend
    from robosandbox.tasks.loader import load_builtin_task

    backend = getattr(args, "sim_backend", "newton")

    try:
        task = load_builtin_task(args.task)
    except FileNotFoundError as e:
        print(f"[eval] {e}", file=sys.stderr)
        return 2

    if backend not in task.supported_backends:
        print(
            f"[eval] task {args.task!r} does not list '{backend}' as a supported backend "
            f"(supported: {', '.join(task.supported_backends)})",
            file=sys.stderr,
        )
        return 2

    try:
        policy = load_policy(Path(args.policy))
    except (ImportError, FileNotFoundError, ValueError) as e:
        print(f"[eval] failed to load policy: {e}", file=sys.stderr)
        return 2

    print(f"[eval] task:          {task.name}")
    print(f"[eval] policy:        {args.policy}")
    print(f"[eval] sim_backend:   {backend}")

    # ---- MuJoCo: single world, RGB available ----------------------------
    if backend == "mujoco":
        try:
            sim = create_sim_backend(
                "mujoco", render_size=(240, 320), camera="scene"
            )
        except (ImportError, ValueError) as e:
            print(f"[eval] failed to create MuJoCo backend: {e}", file=sys.stderr)
            return 2
        sim.load(task.scene)
        try:
            result = run_policy(
                sim, policy,
                max_steps=args.max_steps,
                success=task.success,
            )
        finally:
            sim.close()
        ok = result["success"]
        verdict = "success" if ok else ("failure" if ok is False else "unknown")
        print(f"[eval] verdict:       {verdict}")
        print(f"[eval] steps:         {result['steps']}")
        return 0 if ok else 1

    # ---- Newton: N parallel worlds, state-only --------------------------
    world_count: int = args.world_count
    print(f"[eval] world_count:   {world_count}")
    print(f"[eval] device:        {args.device}")
    print(f"[eval] max_steps:     {args.max_steps}")

    image_keys = _policy_image_inputs(policy)
    if image_keys:
        print(
            f"[eval] WARNING: policy expects image inputs ({', '.join(sorted(image_keys))}) "
            f"but Newton is state-only — it will receive zero-image frames and "
            f"likely produce garbage actions. Use --sim-backend mujoco for vision policies.",
            file=sys.stderr,
        )

    try:
        sim = create_sim_backend(
            "newton",
            render_size=(240, 320),
            camera="scene",
            viewer="null",
            device=args.device,
            world_count=world_count,
        )
        sim.load(task.scene)
    except (ImportError, ValueError) as e:
        print(f"[eval] failed to create Newton backend: {e}", file=sys.stderr)
        return 2

    try:
        result = run_eval_parallel(
            sim,
            policy,
            max_steps=args.max_steps,
            success=task.success,
            settle_steps=args.settle_steps,
        )
    finally:
        sim.close()

    n = result["n_worlds"]
    s = result["successes"]
    rate_pct = result["rate"] * 100.0
    wall = result["wall"]
    throughput = result["throughput"]

    print(f"[eval] successes:     {s} / {n}  ({rate_pct:.1f}%)")
    print(f"[eval] steps:         {result['steps']}")
    print(f"[eval] wall:          {wall:.1f}s")
    print(f"[eval] throughput:    {throughput:,.0f} env-steps/s")
    return 0 if s > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
