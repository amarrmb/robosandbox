# Compared to other tools

Honest table. Where we lose, we lose. The framing on this page is
*"which tool fits which job,"* not *"why we're best."*

## At a glance

| | RoboSandbox | LeRobot | IsaacLab | RoboCasa | robosuite |
|---|---|---|---|---|---|
| **Primary purpose** | Eval contract for manipulation policies | Train + share manipulation policies | Large-scale GPU sim + RL | Large kitchen-scene benchmark | Modular MuJoCo task env |
| **Trains policies?** | No (out of scope) | Yes (ACT, Diffusion Policy, π0, …) | Yes (RL) | No | Indirect (env only) |
| **Eval contract** | Wilson CI, provenance match, spatial breakdown, 7 enforced invariants | Built into trainers; no separate contract | Per-task; varies | Per-task; varies | Per-task; varies |
| **Parallel sim** | Newton, 1–1024+ worlds (state + opt-in batched RGB) | None | Isaac Gym, 1000s of worlds | Isaac Sim under the hood | None (single-world MuJoCo) |
| **Cross-sim eval** | Yes — same task spec runs in MuJoCo *and* Newton | No | No | No | No |
| **Checkpoint-agnostic** | LeRobot today; `Policy` protocol open for others | LeRobot-native | Isaac-native | LeRobot bindings | numpy/torch agnostic |
| **Bring your own arm/task** | YAML + URDF/MJCF; full sidecar schema | Limited (LeRobot-shaped datasets) | Yes (USD) | Limited (Kitchen) | Yes (Python) |
| **Real-robot path** | Stub (`RealRobotBackend` Protocol); SO-101 in progress | Yes (LeRobot-native) | Limited | No | No |
| **Photorealism** | No | N/A (training framework) | Yes (RTX) | Yes (Omniverse) | No |
| **Install footprint** | ~200 MB (core), ~2 GB (+ lerobot) | ~2 GB | ~30 GB (Omniverse) | ~30 GB (Omniverse) | ~200 MB |

## When to pick what

**Pick RoboSandbox when** you want to compare two policies on the
same task with numbers you can defend. The provenance check + Wilson
CI + spatial breakdown is the whole product. Cross-sim eval (Newton ↔
classical MuJoCo) is the only place this exists today as a single
command.

**Pick LeRobot when** you want to *train* a policy and share it on
the Hub. RoboSandbox doesn't train; it consumes LeRobot checkpoints.
The two compose: train in LeRobot, eval here.

**Pick IsaacLab when** you need RTX rendering, USD scenes, large
multi-robot setups, or RL training at full Omniverse scale. The
install is heavy and the API surface is large; don't pick it for a
single-arm pick eval.

**Pick RoboCasa when** the kitchen / household-scene benchmark is
what you actually want to score against. It's a benchmark, not a
sandbox.

**Pick robosuite when** you want a flexible Python-defined MuJoCo
task and you don't need parallelism, eval contracts, or a real-robot
path.

## Where RoboSandbox loses

- **No photorealism.** MuJoCo + Newton, no rendering tricks. If your
  policy needs realistic textures or lighting, this is the wrong tool.
- **Single robot per scene.** Multi-robot is not in scope.
- **Real-robot path is unfinished.** SO-101 backend is a stub today
  (see [sim-to-real handoff](tutorials/sim-to-real-handoff.md)).
- **Policy-framework support is LeRobot-only.** The `Policy` protocol
  is framework-agnostic by design but the only adapter shipped is
  `LeRobotPolicyAdapter`. Diffusion Policy / Octo / π0 need a thin
  user-side wrapper.
- **No training.** Out of scope, full stop. Train elsewhere, evaluate
  here.

## Where the comparison gets murky

The RL track (`experimental/newton-eval`) does train policies via
`robo-sandbox train`. That's intentionally off the public thesis —
see [The RL track](concepts/rl-track.md). If you're comparing
RoboSandbox to IsaacLab specifically for RL, the honest answer today
is: *use IsaacLab for RL training; use RoboSandbox to eval the
resulting checkpoint with a comparable contract.* The two aren't
substitutes.
