# Compared to other tools

Where this fits next to the things it gets compared to.

## At a glance

| | RoboSandbox | LeRobot | IsaacLab | RoboCasa | robosuite |
|---|---|---|---|---|---|
| Primary purpose | Eval contract for manipulation policies | Train + share manipulation policies | Large-scale GPU sim + RL | Kitchen-scene benchmark | Modular MuJoCo task env |
| Trains policies | No (out of scope) | Yes (ACT, Diffusion Policy, π0, …) | Yes (RL) | No | Indirect (env only) |
| Eval contract | Wilson CI, provenance match, spatial breakdown, 7 enforced invariants | Built into trainers; no separate contract | Per-task; varies | Per-task; varies | Per-task; varies |
| Parallel sim | Newton, 1–1024+ worlds (state + opt-in batched RGB) | None | Isaac Gym, 1000s of worlds | Isaac Sim under the hood | None (single-world MuJoCo) |
| Cross-sim eval | Yes — same task spec runs in MuJoCo *and* Newton | No | No | No | No |
| Real-robot path | Stub (`RealRobotBackend` Protocol); SO-101 in progress | Yes (LeRobot-native) | Limited | No | No |
| Photorealism | No | N/A | Yes (RTX) | Yes (Omniverse) | No |
| Install footprint | ~200 MB core, ~2 GB +lerobot | ~2 GB | ~30 GB (Omniverse) | ~30 GB (Omniverse) | ~200 MB |

## When to pick what

**RoboSandbox** — comparing two policies on the same task with numbers you can defend. The provenance check + Wilson CI + spatial breakdown is the whole product. Cross-sim eval (Newton ↔ classical MuJoCo) is the only place this exists today as a single command.

**LeRobot** — training a policy and sharing it on the Hub. RoboSandbox doesn't train; it consumes LeRobot checkpoints. The two compose: train in LeRobot, evaluate here.

**IsaacLab** — RTX rendering, USD scenes, multi-robot setups, RL training at full Omniverse scale. The install is heavy and the API surface is large; not the right tool for a single-arm pick eval.

**RoboCasa** — the kitchen / household-scene benchmark. It's a benchmark, not a sandbox.

**robosuite** — flexible Python-defined MuJoCo task without parallelism, eval contracts, or a real-robot path.

## What this isn't

- **Photorealistic.** MuJoCo + Newton, no rendering tricks. Wrong tool if your policy needs realistic textures or lighting.
- **Multi-robot.** Single robot per scene. Multi-robot is not in scope.
- **A real-robot stack.** The SO-101 backend is a skeleton today (see [sim-to-real handoff](tutorials/sim-to-real-handoff.md)).
- **Drop-in for arbitrary policy frameworks.** `LeRobotPolicyAdapter` is what's shipped. Diffusion Policy / Octo / π0 each need a thin user-side wrapper around the `Policy` protocol.
- **A training framework.** Out of scope. Train elsewhere, evaluate here.

## Where the comparison gets murky

The RL track on `experimental/newton-eval` does train policies via `robo-sandbox train`. That sits intentionally off the public thesis (see [The RL track](concepts/rl-track.md)). For an IsaacLab-vs-RoboSandbox-for-RL comparison today, the answer is: train in IsaacLab, evaluate here against a comparable contract. They aren't substitutes.
