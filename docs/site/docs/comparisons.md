# Compared to Other Tools

The robotics tooling space is crowded, and most of the tools in it
solve different problems despite overlapping vocabularies. This page
sets RoboSandbox next to four tools it gets compared to most often,
calls out what each one is actually for, and is up front about what
RoboSandbox doesn't do.

## What each tool is for

**LeRobot** is a training framework. You bring a dataset and a policy
recipe (ACT, Diffusion Policy, π0, …) and it produces a checkpoint.
RoboSandbox does not train; it consumes LeRobot checkpoints. The two
compose cleanly: train in LeRobot, evaluate in RoboSandbox.

**IsaacLab** is a large-scale GPU simulator for RL training and
photorealistic rendering. The install is heavy, the API surface is
large, and the value is at the high end of fidelity and parallelism.
RoboSandbox is a much smaller piece — wrong tool for a 30 GB
Omniverse install if all you need is a single-arm pick eval.

**RoboCasa** is a kitchen and household-scene benchmark built on top
of Isaac Sim. It is a benchmark, not a sandbox: you score against its
scenes, you don't author your own.

**robosuite** is a flexible Python-defined MuJoCo task environment.
No parallelism, no eval contract, no real-robot path, but the Python
authoring is direct.

## At a glance

| | RoboSandbox | LeRobot | IsaacLab | RoboCasa | robosuite |
|---|---|---|---|---|---|
| Primary purpose | Eval contract for manipulation policies | Train + share manipulation policies | Large-scale GPU sim + RL | Kitchen-scene benchmark | Modular MuJoCo task env |
| Trains policies | No (out of scope) | Yes (ACT, Diffusion Policy, π0, …) | Yes (RL) | No | Indirect (env only) |
| Eval contract | Wilson CI, provenance match, spatial breakdown, seven enforced invariants | Built into trainers; no separate contract | Per-task; varies | Per-task; varies | Per-task; varies |
| Parallel sim | Newton, 1–1024+ worlds (state + opt-in batched RGB) | None | Isaac Gym, 1000s of worlds | Isaac Sim under the hood | None (single-world MuJoCo) |
| Cross-sim eval | Yes — same task spec runs in MuJoCo and Newton | No | No | No | No |
| Real-robot path | Stub Protocol; SO-101 skeleton | Yes (LeRobot-native) | Limited | No | No |
| Photorealism | No | N/A | Yes (RTX) | Yes (Omniverse) | No |
| Install footprint | ~200 MB core, ~2 GB +lerobot | ~2 GB | ~30 GB (Omniverse) | ~30 GB (Omniverse) | ~200 MB |

## What RoboSandbox Doesn't Do

We do not train policies. The eval CLI, the recorder, the export
path, and the sim backends are all there to score and inspect a
checkpoint somebody else trained — there is no built-in trainer and
no plan to add one.

We do not render photorealistic frames. MuJoCo and Newton both render,
but neither is RTX or Omniverse-grade. If your policy needs realistic
textures, lighting, or material reflections, a heavier stack is the
right tool.

We do not support multi-robot scenes. One robot per scene, today.

We do not ship adapters for Diffusion Policy, Octo, or π0. The
`Policy` protocol is framework-agnostic, and writing a wrapper around
those frameworks is a few hours of glue, but those wrappers are not
in the box.

We do not drive real hardware. `RealRobotBackend` is a Protocol with
a software skeleton, and the SO-101 backend tracks commanded state in
memory. Wiring a real motor bus, a real camera, and the calibration
that goes around them is left to the user — see
[sim-to-real handoff](tutorials/sim-to-real-handoff.md) for what the
contract expects.

## Where the Comparison Gets Murky

The RL track on `experimental/newton-eval` does train policies via
`robo-sandbox train`. That is intentionally off the public thesis —
see [the RL track concept page](concepts/rl-track.md). If you are
comparing RoboSandbox to IsaacLab specifically for RL training, the
honest answer is: train in IsaacLab, evaluate here against a
comparable contract. The two are not substitutes.
