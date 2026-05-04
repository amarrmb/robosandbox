# Where Things Are

Some of what RoboSandbox does today lives on `main`, and some lives on
the `experimental/newton-eval` branch. The split isn't arbitrary, but
it's not always obvious where to look. This page maps each thing to
the page that covers it and the branch it currently lives on.

## On `main`

The viewer, the scripted demo, the LeRobot export path, and policy
replay against a public LeRobot checkpoint are all on `main`. So is
the `RealRobotBackend` Protocol with its software skeleton, and the
SO-101 backend skeleton — neither of which drives real hardware yet,
but they do let you run observation-and-step skills against a backend
that isn't MuJoCo.

The on-ramp:

- [Quickstart](quickstart.md) — install, open the viewer, record one
  episode in five minutes.
- [LeRobot export](tutorials/lerobot-export.md) — turn a recorded run
  into a LeRobot v3 parquet dataset.
- [LeRobot policy replay](tutorials/lerobot-policy-replay.md) — wrap
  a public ACT checkpoint with `LeRobotPolicyAdapter` and run it
  through `run_policy` under cross-embodiment mismatch.
- [Sim-to-real handoff](tutorials/sim-to-real-handoff.md) — what the
  `RealRobotBackend` Protocol expects and what's still missing for a
  real arm.
- [Bring your own robot](guides/bring-your-own-robot.md) /
  [object](guides/bring-your-own-object.md) /
  [task](guides/bring-your-own-task.md) — the URDF, mesh, and YAML
  authoring paths.

## On `experimental/newton-eval`

The eval CLI (`robo-sandbox eval`), the structured eval JSON (Wilson
CI, spatial breakdown, provenance), the Newton GPU backend, the
cross-sim transfer harness, and the experimental world-model backend
all live on this branch. They land on `main` together with the
eval-and-recording-hygiene PR.

The RL training pieces stay on the experimental branch and are not
planned to land on `main`. Training models is intentionally out of
scope for the project; the `train` CLI, `rl/` module, eval-log
lineage, and `results` subcommands exist on the branch because that's
where the Newton parallel sim integration was built and validated.

What's there:

- [Train ACT and eval it](tutorials/train-act-and-eval.md) — the
  recipe end-to-end, including the seven gotchas that show up the
  first time you run it.
- [Iterating on a policy](guides/iterating-on-a-policy.md) — using
  the eval JSON's `spatial_breakdown` to find where a policy fails
  and target new demos.
- [The eval contract](concepts/the-eval-contract.md) — the schema
  and the seven invariants enforced in code.
- [The RL track](concepts/rl-track.md) — what works (reach 1024w at
  100%, USB-A insertion 96.5% at 1024w, 100% cross-sim at n=32) and
  what does not (warm-started PPO from BC on contact-rich pick).
- [Eval for world models](concepts/eval-for-world-models.md) and
  [World model as a sim backend](tutorials/world-model-as-sim-backend.md)
  — the slot for learned dynamics propagators and what it would
  take to plug in V-JEPA-2, Cosmos, or DreamerV3.

## On the IL ↔ RL Bridge

The natural mental model is to train an ACT policy in MuJoCo, distill
it to a state-only MLP, warm-start PPO in Newton, and fine-tune.
That path is wired but currently fails on contact-rich pick because
of a structural Newton-vs-classical-MuJoCo gravity-settling gap. The
distilled MLP scores 34% in MuJoCo and 0% in Newton; the PPO
fine-tune then has no positive reward signal and wipes the prior.
Concretely:

- The IL track ends at `robo-sandbox eval` in MuJoCo. That part
  works.
- The RL track is from-scratch curriculum PPO on tasks Newton solves
  cleanly — reach (1024 worlds, 100% success) and connector insertion
  (a 4 mm → 2 mm → 1.5 mm USB-A curriculum, 100% at 64 worlds, 96.5%
  at 1024 worlds, 100% cross-sim on classical MuJoCo at n=32).

The two tracks are siblings, not a sequential pipeline. Don't expect
to chain them through warm-start on contact-rich pick today.

The structural cause and the reproduction commands are in
[The RL track](concepts/rl-track.md). The diagnostics are in
`scripts/probe_*` on `experimental/newton-eval`.
