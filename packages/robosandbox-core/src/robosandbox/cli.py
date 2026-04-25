"""`robo-sandbox` CLI entry point."""

from __future__ import annotations

import argparse
import math
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
    exp_p.add_argument(
        "src",
        help=(
            "Source: either a single episode dir (containing events.jsonl) "
            "or a parent dir containing many episode subdirs — auto-detected. "
            "Use --src repeatedly to combine an explicit list."
        ),
    )
    exp_p.add_argument("dst", help="Destination LeRobot dataset dir")
    exp_p.add_argument("--task", default=None, help="Override task string")
    exp_p.add_argument("--fps", type=int, default=30)
    exp_p.add_argument(
        "--src",
        dest="extra_src",
        action="append",
        default=[],
        help="Additional source episode dirs (repeatable). Combined with the positional src.",
    )

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
    eval_p.add_argument("--n-trials", type=int, default=1,
                        help="MuJoCo only: repeat the eval N times. Newton uses --world-count.")
    eval_p.add_argument("--output", default=None,
                        help="Write structured eval result as JSON to this path.")
    eval_p.add_argument("--seed", type=int, default=None,
                        help="Seed for any randomization (item 3 will use this).")
    eval_p.add_argument("--reload-policy", action=argparse.BooleanOptionalAction, default=True,
                        help=(
                            "Reload the policy fresh per trial (default true) so per-trial "
                            "outcomes are deterministic and independent of trial order. The "
                            "alternative (--no-reload-policy) loads once and only calls "
                            "policy.reset() between trials, which is faster but leaks state "
                            "(BatchNorm running buffers, register_buffer tensors, RNG, "
                            "compile/autograd cache) across trials — observed seed-level "
                            "outcomes change with trial order. Reload cost is ~1-2 s/trial."
                        ))
    eval_p.add_argument("--newton-rgb", action=argparse.BooleanOptionalAction, default=None,
                        help=(
                            "Newton only: raytrace per-world RGB via SensorTiledCamera. "
                            "Default = auto (on iff the policy declares image inputs). "
                            "--no-newton-rgb forces zero frames (faster, useful only for "
                            "state-only policies)."
                        ))
    eval_p.add_argument("--action-repeat", type=int, default=1,
                        help=(
                            "Hold each policy action for N sim steps. Set this "
                            "to sim_dt/dataset_dt (e.g. 6 when training data "
                            "was recorded at 30 fps and sim_dt=0.005s = 200 Hz) "
                            "so the policy sees the same temporal cadence at "
                            "eval as it did during training. Default 1 (call "
                            "policy every step) is correct only when training "
                            "and eval cadences match."
                        ))

    sc_p = sub.add_parser(
        "sim-check",
        help="Drive the same policy through MuJoCo and Newton; report agreement",
    )
    sc_p.add_argument("--task", required=True)
    sc_p.add_argument("--policy", required=True)
    sc_p.add_argument("--max-steps", type=int, default=600)
    sc_p.add_argument("--settle-steps", type=int, default=100)
    sc_p.add_argument("--joint-tol-rad", type=float, default=0.087,
                      help="Per-joint tolerance for end-state agreement (~5 deg).")
    sc_p.add_argument("--xy-tol-m", type=float, default=0.005,
                      help="Per-object xy tolerance for end-state agreement (5 mm).")
    sc_p.add_argument("--device", default="cuda:0")
    sc_p.add_argument("--output", default=None,
                      help="Write the full report as JSON to this path.")

    cmp_p = sub.add_parser(
        "compare",
        help="Statistical comparison of two eval JSON outputs (two-proportion z-test)",
    )
    cmp_p.add_argument("a", help="Path to first eval JSON (from `eval --output`)")
    cmp_p.add_argument("b", help="Path to second eval JSON")
    cmp_p.add_argument("--alpha", type=float, default=0.05,
                       help="Significance threshold for the difference (default 0.05)")

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

        from robosandbox.export.lerobot import export_episode, export_episodes

        # Resolve sources: positional + any --src flags. If a single source
        # is supplied and it directly contains events.jsonl, use the
        # single-episode fast path. Otherwise the source is treated as a
        # parent directory and every immediate subdir with events.jsonl
        # becomes one episode in the combined dataset.
        positional = Path(args.src)
        extras = [Path(p) for p in (getattr(args, "extra_src", None) or [])]
        if not extras and (positional / "events.jsonl").exists():
            sources = [positional]
        elif extras:
            sources = [positional, *extras]
        else:
            # Parent-dir auto-detect: any immediate subdir with events.jsonl.
            subs = sorted(
                p for p in positional.iterdir()
                if p.is_dir() and (p / "events.jsonl").exists()
            )
            if not subs:
                print(
                    f"[export-lerobot] no episodes found under {positional} "
                    f"(no events.jsonl in {positional} and no immediate "
                    f"subdirs containing it)",
                    file=sys.stderr,
                )
                return 2
            sources = subs

        if len(sources) == 1:
            out = export_episode(sources[0], Path(args.dst), task=args.task, fps=args.fps)
            print(f"Exported LeRobot dataset to: {out}  (1 episode)")
        else:
            out = export_episodes(sources, Path(args.dst), task=args.task, fps=args.fps)
            print(f"Exported LeRobot dataset to: {out}  ({len(sources)} episodes)")
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
            "  Note: vision policies (ACT, Diffusion) work on either backend. "
            "Newton auto-renders per-world RGB via SensorTiledCamera when the "
            "policy declares image inputs (override with --no-newton-rgb)."
        )
        return 0
    elif args.cmd == "eval":
        return _eval_parallel_cli(args)
    elif args.cmd == "compare":
        return _compare_cli(args)
    elif args.cmd == "sim-check":
        return _sim_check_cli(args)
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


def _print_eval_summary(summary: dict) -> None:
    """Pretty-print a finished EvalSummary dict to stdout."""
    n = int(summary["n_trials"])
    s = int(summary["successes"])
    rate_pct = float(summary["rate"]) * 100.0
    lo_pct = float(summary["ci_low"]) * 100.0
    hi_pct = float(summary["ci_high"]) * 100.0
    ci_pct = int(round(float(summary["ci_level"]) * 100))
    wall = float(summary["wall_seconds"])
    thr = float(summary["throughput_env_steps_per_s"])
    print(f"[eval] successes:     {s} / {n}  ({rate_pct:.1f}%)")
    print(f"[eval] {ci_pct}% CI (Wilson): [{lo_pct:.1f}%, {hi_pct:.1f}%]")
    print(f"[eval] steps:         {summary['steps']}")
    print(f"[eval] wall:          {wall:.1f}s")
    print(f"[eval] throughput:    {thr:,.0f} env-steps/s")


def _maybe_write_json(path: str | None, summary: dict) -> None:
    """Write the EvalSummary as JSON when ``--output`` was supplied."""
    if not path:
        return
    import json
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote JSON:    {out}")


def _finalize_eval(
    args: argparse.Namespace,
    *,
    task_name: str,
    sim_backend: str,
    successes: int,
    n_trials: int,
    success_per_trial: list[bool],
    per_trial_details: list[dict] | None,
    steps: int,
    wall_seconds: float,
    throughput: float,
) -> int:
    """Build summary, print it, optionally write JSON, return exit code."""
    from robosandbox.eval import summarise_eval
    summary = summarise_eval(
        task=task_name,
        policy=str(args.policy),
        sim_backend=sim_backend,
        successes=successes,
        n_trials=n_trials,
        success_per_trial=success_per_trial,
        per_trial_details=per_trial_details,
        provenance=_build_provenance(str(args.policy)),
        steps=steps,
        wall_seconds=wall_seconds,
        throughput=throughput,
    )
    _print_eval_summary(summary)
    _maybe_write_json(getattr(args, "output", None), summary)
    return 0 if successes > 0 else 1


def _build_provenance(policy_path: str) -> dict:
    """Collect the bits that pin down an eval's reproducibility:

    - SHA-256 of the model weights (``model.safetensors``) — content hash
      that survives renames and path changes; two evals with matching
      hashes used the same checkpoint bytes.
    - robosandbox + lerobot + mujoco version strings (``importlib.metadata``).
    - Git rev of the working repo, with a ``-dirty`` suffix when there are
      uncommitted changes — so a "clean" rev guarantees the code is the
      committed version.

    All fields are best-effort: missing dependencies, non-git checkouts,
    or a missing model file just produce ``None``/absent keys instead of
    failing the eval.
    """
    import hashlib
    import subprocess
    from pathlib import Path
    p = Path(policy_path)
    prov: dict = {"policy_path": str(p)}

    # Checkpoint content hash. Use the lerobot canonical filename when
    # present; fall back to events.jsonl for replay policies.
    weight_file = None
    for candidate in ("model.safetensors", "events.jsonl"):
        f = p / candidate if p.is_dir() else p
        if f.exists() and f.is_file():
            weight_file = f
            break
    if weight_file is not None:
        h = hashlib.sha256()
        with weight_file.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        prov["checkpoint_file"] = weight_file.name
        prov["checkpoint_sha256"] = h.hexdigest()
        prov["checkpoint_bytes"] = weight_file.stat().st_size

    # Versions of the libraries that produced the eval result.
    try:
        from importlib.metadata import PackageNotFoundError, version
        for pkg in ("robosandbox-core", "lerobot", "mujoco", "torch", "warp-lang"):
            try:
                prov[f"{pkg}_version"] = version(pkg)
            except PackageNotFoundError:
                pass
    except ImportError:
        pass

    # Git rev of the robosandbox checkout (with -dirty suffix for any
    # uncommitted changes — anything but a clean tag is a flag for "the
    # eval was run against modified code").
    try:
        repo = Path(__file__).resolve().parent.parent.parent
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True, timeout=2,
        )
        if rev.returncode == 0:
            sha = rev.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=repo,
                capture_output=True, text=True, timeout=2,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            prov["robosandbox_git_rev"] = sha
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return prov


def _sim_check_cli(args: argparse.Namespace) -> int:
    """Run the cross-backend agreement check; print the report."""
    import json
    from pathlib import Path

    from robosandbox.eval.sim_check import report_to_dict, run_sim_check

    rep = run_sim_check(
        task_name=args.task,
        policy_path=args.policy,
        max_steps=int(args.max_steps),
        settle_steps=int(args.settle_steps),
        joint_tol_rad=float(args.joint_tol_rad),
        xy_tol_m=float(args.xy_tol_m),
        device=args.device,
    )

    print(f"[sim-check] task:           {rep.task}")
    print(f"[sim-check] policy:         {rep.policy}")
    print(f"[sim-check] tolerances:     joint <= {rep.tolerances['joint_rad']:.3f} rad, xy <= {rep.tolerances['xy_m'] * 1000:.1f} mm")

    if "error" in rep.mujoco:
        print(f"[sim-check] MuJoCo FAILED: {rep.mujoco['error']}", file=sys.stderr)
    else:
        print(f"[sim-check] MuJoCo:         success={rep.mujoco['success']}  steps={rep.mujoco['steps']}")
    if "error" in rep.newton:
        print(f"[sim-check] Newton FAILED:  {rep.newton['error']}", file=sys.stderr)
    elif rep.newton:
        print(f"[sim-check] Newton:         success={rep.newton['success']}  steps={rep.newton['steps']}")

    if rep.verdict != "RUN_FAILED":
        print(f"[sim-check] outcome match:  {rep.outcome_match}")
        print(f"[sim-check] joint max diff: {rep.joint_max_abs_diff_rad:.4f} rad")
        print(f"[sim-check] object max xy:  {rep.object_max_xy_diff_m * 1000:.2f} mm")
        for oid, d in sorted(rep.object_diffs.items()):
            print(f"[sim-check]   {oid}: {d * 1000:.2f} mm")
    print(f"[sim-check] verdict:        {rep.verdict}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_to_dict(rep), indent=2))
        print(f"[sim-check] wrote JSON:     {out}")

    # Exit code: 0 on PASS, 1 on STATE_DRIFT (the backends agreed on outcome
    # but drifted in state — a softer failure), 2 on OUTCOME_MISMATCH or
    # RUN_FAILED (the load-bearing claim broke).
    if rep.verdict == "PASS":
        return 0
    if rep.verdict == "STATE_DRIFT":
        return 1
    return 2


def _compare_cli(args: argparse.Namespace) -> int:
    """Two-proportion z-test comparison of two EvalSummary JSON files."""
    import json
    from pathlib import Path

    from robosandbox.eval import proportion_z_test

    try:
        a = json.loads(Path(args.a).read_text())
        b = json.loads(Path(args.b).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[compare] failed to read input: {e}", file=sys.stderr)
        return 2

    for label, doc in [("A", a), ("B", b)]:
        for required in ("successes", "n_trials", "task", "policy", "sim_backend"):
            if required not in doc:
                print(f"[compare] {label} is missing required field {required!r}", file=sys.stderr)
                return 2
    if a["task"] != b["task"]:
        print(
            f"[compare] WARNING: comparing across tasks ({a['task']} vs {b['task']}). "
            f"The z-test result is mathematically valid but semantically suspect.",
            file=sys.stderr,
        )

    s_a, n_a = int(a["successes"]), int(a["n_trials"])
    s_b, n_b = int(b["successes"]), int(b["n_trials"])
    rate_a = (s_a / n_a) if n_a else 0.0
    rate_b = (s_b / n_b) if n_b else 0.0
    z, p_value = proportion_z_test(s_a, n_a, s_b, n_b)
    significant = p_value < float(args.alpha)
    if rate_a > rate_b:
        winner = "A"
    elif rate_b > rate_a:
        winner = "B"
    else:
        winner = "tie"

    print(f"A: {a['policy']}  ({a['sim_backend']})")
    print(f"   successes: {s_a}/{n_a}  ({rate_a * 100:.1f}%)  CI: [{a.get('ci_low', 0) * 100:.1f}%, {a.get('ci_high', 1) * 100:.1f}%]")
    print(f"B: {b['policy']}  ({b['sim_backend']})")
    print(f"   successes: {s_b}/{n_b}  ({rate_b * 100:.1f}%)  CI: [{b.get('ci_low', 0) * 100:.1f}%, {b.get('ci_high', 1) * 100:.1f}%]")
    print(f"diff:       {(rate_a - rate_b) * 100:+.1f}pp")
    print(f"z:          {z:+.3f}")
    print(f"p-value:    {p_value:.4g}")
    if winner == "tie":
        print(f"verdict:    tie (rates equal)")
    elif significant:
        print(f"verdict:    {winner} > other  (p<{args.alpha:g}, significant)")
    else:
        print(f"verdict:    {winner} > other  but NOT significant at α={args.alpha:g}")
    # Exit 0 either way — `compare` is a report tool, not a gate. Callers
    # who want a gate should check the printed p-value themselves.
    return 0


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
        import time as _time

        from robosandbox.eval import summarise_eval
        from robosandbox.tasks.randomize import jitter_scene

        n_trials = max(1, int(getattr(args, "n_trials", 1) or 1))
        base_seed = getattr(args, "seed", None)
        randomize_active = bool(task.randomize) and n_trials > 1
        print(f"[eval] n_trials:      {n_trials}")
        if task.randomize and n_trials == 1:
            print(
                "[eval] note:         task has randomize spec but --n-trials=1; "
                "using base scene (seed=0). Pass --n-trials N to randomize.",
                file=sys.stderr,
            )
        if randomize_active:
            seed_origin = "user-provided" if base_seed is not None else "auto-derived"
            print(f"[eval] randomize:     {sorted(task.randomize.keys())}  ({seed_origin} seeds)")
        from robosandbox.tasks.runner import criterion_target_object
        per_trial_details: list[dict] = []
        success_per_trial: list[bool] = []
        target_obj_id = criterion_target_object(task.success)
        total_steps = 0
        t0 = _time.time()
        for trial in range(n_trials):
            try:
                sim = create_sim_backend(
                    "mujoco", render_size=(240, 320), camera="scene"
                )
            except (ImportError, ValueError) as e:
                print(f"[eval] failed to create MuJoCo backend: {e}", file=sys.stderr)
                return 2
            # Per-trial seed: 0 returns the base scene unchanged, so we shift
            # the trial index by 1 in the auto-derived path. With --seed S,
            # trial t uses S + t (still skipping 0 -> identity).
            if randomize_active:
                trial_seed = (int(base_seed) + trial + 1) if base_seed is not None else (trial + 1)
                trial_scene = jitter_scene(task.scene, task.randomize, trial_seed)
            else:
                trial_scene = task.scene
            sim.load(trial_scene)
            # Match the settle the recorder did before its first frame —
            # without it the policy sees mid-fall, OOD vs training.
            for _ in range(int(getattr(args, "settle_steps", 100) or 0)):
                sim.step()
            if getattr(args, "reload_policy", True):
                trial_policy = load_policy(Path(args.policy))
            else:
                trial_policy = policy
            initial_cube_pose = sim.get_object_pose(target_obj_id) if target_obj_id else None
            initial_cube_z = initial_cube_pose.xyz[2] if initial_cube_pose else None
            peak_lift_mm = 0.0
            min_ee_dist_mm = float("inf")
            def _track(obs: object, action: object,
                       _z0=initial_cube_z, _oid=target_obj_id) -> None:
                nonlocal peak_lift_mm, min_ee_dist_mm
                if _oid is None or _z0 is None:
                    return
                p = sim.get_object_pose(_oid)
                if p is None:
                    return
                lift_mm = (p.xyz[2] - _z0) * 1000.0
                if lift_mm > peak_lift_mm:
                    peak_lift_mm = lift_mm
                ee_xyz = obs.ee_pose.xyz  # type: ignore[attr-defined]
                d_mm = math.sqrt(sum((a - b) ** 2 for a, b in zip(ee_xyz, p.xyz))) * 1000.0
                if d_mm < min_ee_dist_mm:
                    min_ee_dist_mm = d_mm
            try:
                # Reset replay-style policies so each trial starts at step 0.
                reset = getattr(trial_policy, "reset", None)
                if callable(reset):
                    reset()
                result = run_policy(
                    sim, trial_policy,
                    max_steps=args.max_steps,
                    success=task.success,
                    action_repeat=int(getattr(args, "action_repeat", 1) or 1),
                    on_step=_track,
                )
            finally:
                sim.close()
            ok = bool(result["success"]) if result["success"] is not None else False
            success_per_trial.append(ok)
            total_steps += int(result["steps"])
            trial_seed_recorded = (
                ((int(base_seed) + trial + 1) if base_seed is not None else (trial + 1))
                if randomize_active else 0
            )
            detail = {
                "trial": trial + 1,
                "seed": trial_seed_recorded,
                "success": ok,
                "success_step": result.get("success_step"),
                "steps": int(result["steps"]),
                "peak_lift_mm": float(peak_lift_mm),
                "min_ee_object_dist_mm": (
                    float(min_ee_dist_mm) if min_ee_dist_mm != float("inf") else None
                ),
            }
            if initial_cube_pose is not None:
                detail["object_id"] = target_obj_id
                detail["object_initial_xyz"] = list(initial_cube_pose.xyz)
                detail["object_initial_quat_xyzw"] = list(initial_cube_pose.quat_xyzw)
            per_trial_details.append(detail)
            if n_trials > 1:
                print(f"[eval]   trial {trial + 1}/{n_trials}: {'success' if ok else 'failure'} ({result['steps']} steps)")
        wall = _time.time() - t0
        successes = sum(success_per_trial)
        return _finalize_eval(
            args, task_name=task.name, sim_backend="mujoco",
            successes=successes, n_trials=n_trials,
            success_per_trial=success_per_trial,
            per_trial_details=per_trial_details,
            steps=total_steps, wall_seconds=wall,
            throughput=(total_steps / wall) if wall > 0 else 0.0,
        )

    # ---- Newton: N parallel worlds, state-only --------------------------
    world_count: int = args.world_count
    print(f"[eval] world_count:   {world_count}")
    print(f"[eval] device:        {args.device}")
    print(f"[eval] max_steps:     {args.max_steps}")

    image_keys = _policy_image_inputs(policy)
    # Auto-enable RGB if the policy needs it; users can force on/off via
    # --newton-rgb / --no-newton-rgb. Camera-on costs ~ms per step but is
    # required for vision policies; off keeps state-only RL fast.
    if getattr(args, "newton_rgb", None) is None:
        enable_camera = bool(image_keys)
    else:
        enable_camera = bool(args.newton_rgb)
    if image_keys and not enable_camera:
        print(
            f"[eval] WARNING: policy expects image inputs ({', '.join(sorted(image_keys))}) "
            f"but --no-newton-rgb was set — frames will be zeros. Drop the flag to "
            f"raytrace per-world RGB via SensorTiledCamera.",
            file=sys.stderr,
        )
    elif enable_camera:
        print(f"[eval] camera:        SensorTiledCamera (1 cam/world @ 240x320)")

    # Per-world randomization: build N different scenes when the task's
    # randomize spec is set. Topology is invariant under jitter_scene so
    # the Newton backend's per-world layout assumptions still hold.
    per_world_scenes = None
    if task.randomize and world_count > 1:
        from robosandbox.tasks.randomize import jitter_scene as _jitter
        base_seed = getattr(args, "seed", None)
        per_world_scenes = [
            _jitter(
                task.scene,
                task.randomize,
                seed=((int(base_seed) + w + 1) if base_seed is not None else (w + 1)),
            )
            for w in range(world_count)
        ]
        seed_origin = "user-provided" if base_seed is not None else "auto-derived"
        print(f"[eval] randomize:     {sorted(task.randomize.keys())}  ({seed_origin} seeds, per-world)")
    elif task.randomize and world_count == 1:
        print(
            "[eval] note:         task has randomize spec but --world-count=1; "
            "using base scene. Increase --world-count to randomize.",
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
            enable_camera=enable_camera,
        )
        sim.load(task.scene, per_world_scenes=per_world_scenes)
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

    return _finalize_eval(
        args, task_name=task.name, sim_backend="newton",
        successes=int(result["successes"]), n_trials=int(result["n_worlds"]),
        success_per_trial=list(result.get("success_per_world", [])),
        per_trial_details=None,  # Newton parallel path doesn't track per-trial pose
        steps=int(result["steps"]), wall_seconds=float(result["wall"]),
        throughput=float(result["throughput"]),
    )


if __name__ == "__main__":
    sys.exit(main())
