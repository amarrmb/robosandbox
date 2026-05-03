"""Render each working scripted skill in MuJoCo, then stitch a single
montage video for the Beat 1 build-in-public post.

Per-clip output: runs/<timestamp>-<name>/video.mp4 (320x240 @ 30fps)
Final montage:  outputs/beat1_montage/skills_montage.mp4 (1920x1080)

Usage (laptop):
    MUJOCO_GL=egl python scripts/record_skills_montage.py
    MUJOCO_GL=egl python scripts/record_skills_montage.py --only pick pour

Each trial that fails is skipped from the montage (not silently — printed).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(
    0, str(Path(__file__).parent.parent / "packages" / "robosandbox-core" / "src")
)

from robosandbox.agent.context import AgentContext
from robosandbox.grasp.analytic import AnalyticTopDown
from robosandbox.motion.ik import DLSMotionPlanner
from robosandbox.perception.ground_truth import GroundTruthPerception
from robosandbox.recorder.local import LocalRecorder
from robosandbox.sim.mujoco_backend import MuJoCoBackend
from robosandbox.tasks.loader import load_builtin_task


@dataclass
class Trial:
    name: str
    caption: str  # human-readable title for the lower-third in the montage
    task: str
    skill_factory: callable
    extra_skills: list = None  # list of (skill_factory, args) to chain after the main skill
    skill_args: dict = None

    def __post_init__(self):
        if self.skill_args is None:
            self.skill_args = {}
        if self.extra_skills is None:
            self.extra_skills = []


RUNS_ROOT = Path("/tmp/skills_montage_runs")


def render_one_clip(trial: Trial, render_size=(480, 640)) -> Path | None:
    """Run one (skill, task) and return the path to the resulting video.mp4 (or None on failure)."""
    print(f"[clip] === {trial.name} === ({trial.caption})")
    if RUNS_ROOT.exists():
        # We let recorder make per-trial timestamped subdirs; only nuke contents on first call
        pass
    try:
        task = load_builtin_task(trial.task)
        sim = MuJoCoBackend(render_size=render_size, camera="scene")
        sim.load(task.scene)
        recorder = LocalRecorder(root=RUNS_ROOT, video_fps=30)

        def _on_step():
            recorder.write_frame(sim.observe(), action=sim.last_action())

        ctx = AgentContext(
            sim=sim,
            perception=GroundTruthPerception(),
            grasp=AnalyticTopDown(),
            motion=DLSMotionPlanner(n_waypoints=160, dt=0.005),
            recorder=recorder,
            on_step=_on_step,
        )
        ep_id = recorder.start_episode(task=task.prompt, metadata={"clip": trial.name})

        # Settle physics. Capture frames during settle to avoid an abrupt start.
        for _ in range(60):
            sim.step()
            recorder.write_frame(sim.observe(), action=sim.last_action())

        skill = trial.skill_factory()
        t0 = time.time()
        result = skill(ctx, **trial.skill_args)
        for fac, args in trial.extra_skills:
            extra = fac()
            result = extra(ctx, **args)
        wall = time.time() - t0

        # Hold a final 1s of frames so the success state is visible
        for _ in range(30):
            sim.step()
            recorder.write_frame(sim.observe(), action=sim.last_action())

        success = bool(getattr(result, "success", False))
        recorder.end_episode(success=success,
                             result={"reason": getattr(result, "reason", "?")})
        sim.close()

        # Find the latest episode dir for this trial
        dirs = sorted(RUNS_ROOT.glob("20*"), reverse=True)
        if not dirs:
            print(f"[clip]   ERROR: no episode dir found")
            return None
        ep_dir = dirs[0]
        video = ep_dir / "video.mp4"
        if not video.exists():
            print(f"[clip]   ERROR: video.mp4 missing in {ep_dir}")
            return None

        # Tag the directory with the trial name for traceability
        tagged = ep_dir.with_name(ep_dir.name + f"-{trial.name}")
        ep_dir.rename(tagged)
        video = tagged / "video.mp4"

        flag = "OK" if success else "fail"
        print(f"[clip]   {flag}  ({wall:.1f}s)  → {video}")
        return video if success else None
    except Exception as e:
        print(f"[clip]   EXCEPTION: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def build_montage(clips: list[tuple[Path, str]], out: Path,
                  fps: int = 30, target_w: int = 1920, target_h: int = 1080,
                  target_clip_sec: float = 6.5) -> int:
    """Build a 1080p montage from per-clip videos with simple title overlays.

    Each clip is scaled+padded to fill 1920x1080, then a lower-third caption
    overlay is drawn for the first 2 seconds of the clip. Clips are concatenated.
    """
    if not clips:
        print("[montage] no clips to stitch")
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)

    # Build complex filter
    inputs = []
    filter_parts = []
    font = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"
    if not Path(font).exists():
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

    for i, (clip, caption) in enumerate(clips):
        # Probe source duration → choose speed so each clip lands ~target_clip_sec
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(clip)],
            capture_output=True, text=True,
        )
        try:
            src_dur = float(probe.stdout.strip())
        except ValueError:
            src_dur = 30.0
        speed = max(1.0, src_dur / target_clip_sec)
        inputs += ["-i", str(clip)]
        # Speed up source so the full skill (approach → completion) plays in
        # ~target_clip_sec. No trim — the success state is at the end and
        # MUST be visible.
        f = (
            f"[{i}:v]"
            f"setpts=PTS/{speed:.3f},"
            f"fps={fps},"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=0x0f172a,"
            f"drawbox=x=0:y=ih-180:w=iw:h=180:color=0x0f172a@0.85:t=fill,"
            f"drawtext=fontfile={font}:text='{caption}':fontcolor=white:fontsize=56:"
            f"x=(w-text_w)/2:y=h-130,"
            f"setsar=1,format=yuv420p[v{i}]"
        )
        filter_parts.append(f)
        print(f"[montage]   {clip.parent.name}: src={src_dur:.1f}s speed={speed:.2f}x → ~{src_dur/speed:.1f}s")

    concat_inputs = "".join(f"[v{i}]" for i in range(len(clips)))
    filter_parts.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=0[v]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-movflags", "+faststart",
        str(out),
    ]
    print(f"[montage] ffmpeg → {out}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[montage] FAIL: {proc.stderr[-2000:]}")
        return proc.returncode
    print(f"[montage] wrote {out}  ({Path(out).stat().st_size/1e6:.1f} MB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--out", type=Path,
                        default=Path("outputs/beat1_montage/skills_montage.mp4"))
    parser.add_argument("--clean", action="store_true",
                        help="Remove RUNS_ROOT before starting")
    args = parser.parse_args()

    if args.clean and RUNS_ROOT.exists():
        shutil.rmtree(RUNS_ROOT)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    from robosandbox.skills.pick import Pick
    from robosandbox.skills.place import PlaceOn
    from robosandbox.skills.drawer import OpenDrawer
    from robosandbox.skills.pour import Pour
    from robosandbox.skills.tap import Tap

    trials = [
        Trial("pick_cube", "Pick — red cube",
              "pick_cube_franka_random", Pick, skill_args=dict(object="red cube")),
        Trial("pick_mug", "Pick — YCB mug",
              "pick_ycb_mug", Pick, skill_args=dict(object="mug")),
        Trial("pick_place", "Pick → Place — red cube on green cube",
              "_experimental_stack_two", Pick, skill_args=dict(object="red cube"),
              extra_skills=[(PlaceOn, dict(target="green cube", height_offset=0.024))]),
        Trial("open_drawer", "Open drawer",
              "open_drawer", OpenDrawer, skill_args=dict(drawer="drawer_a")),
        Trial("pour", "Pick + Pour — soup can into bowl",
              "pour_can_into_bowl", Pick, skill_args=dict(object="tomato soup can"),
              extra_skills=[(Pour, dict(target="bowl"))]),
        Trial("tap", "Tap — red cube",
              "pick_cube_franka_random", Tap, skill_args=dict(object="red cube")),
    ]

    if args.only:
        trials = [t for t in trials if any(s in t.name for s in args.only)]

    print(f"[run] {len(trials)} trials")
    clips: list[tuple[Path, str]] = []
    for tr in trials:
        v = render_one_clip(tr)
        if v is not None:
            clips.append((v, tr.caption))

    print()
    print(f"[run] {len(clips)}/{len(trials)} clips usable")
    if not clips:
        print("[run] no usable clips — aborting montage")
        return 1

    return build_montage(clips, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
