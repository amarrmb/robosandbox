# Eval for world models

The world-model wave is real. World Labs raised $230M then $1B. Yann
LeCun's AMI Labs raised $1.03B at $3.5B. NVIDIA Cosmos has open
checkpoints. Meta's V-JEPA-2 weights are out. The bet across all of
them: learned dynamics will replace classical physics simulators for
robotics.

If that bet pays off — even partially, even just for navigation
before manipulation — then "evaluate a policy in MuJoCo or Newton"
gets a third option: evaluate it in a learned world model.

This page is about what RoboSandbox does in that world.

## The substrate question is not the eval question

These are different layers of the stack:

- **Substrate**: what produces the next observation given an action?
  Classical sim, GPU sim, a learned world model, the real arm.
- **Eval contract**: given a checkpoint and a substrate, what counts
  as success, what's the schema for the result, how do you know two
  numbers are comparable?

World-model labs are building substrates. Nobody is building eval
contracts. Today every world-model paper hand-rolls its own success
criteria, its own statistics, its own provenance. When the field
eventually has to compare two world models or two policies trained on
two world models, *somebody* has to ship the contract layer.

That's our shape. We are not competing with World Labs. We are the
layer that should sit on top when their substrate is ready.

## What "world model as a `SimBackend`" means in this codebase

The `SimBackend` Protocol is small:

```python
class SimBackend(Protocol):
    def load(self, scene: Scene) -> None: ...
    def reset(self) -> None: ...
    def step(self, target_joints, gripper) -> None: ...
    def observe(self) -> Observation: ...
    @property
    def n_dof(self) -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    def close(self) -> None: ...
```

Anything that can answer *"given a state and an action, what's the
next state"* fits this slot. Classical solvers fit (MuJoCo, Newton).
Learned dynamics models fit too — they just produce the next
observation differently.

The branch ships a working `WorldModelBackend` that takes a
`WorldModelPredictor` and runs it through the same `robo-sandbox eval`
pipeline as MuJoCo or Newton. See
**[World model as a sim backend](../tutorials/world-model-as-sim-backend.md)**
for the wiring and a smoke-test result. The reference predictor is
intentionally trivial — the wedge is the *slot*, not a world-model
implementation.

## What it would take to plug in a real world model

Each of the named candidates has a different integration shape.
Skimming, in rough order of effort:

| Model | Public weights | Action-conditional? | Manipulation-relevant? | Integration shape |
|---|---|---|---|---|
| **V-JEPA-2** (Meta) | yes | latent-space, unclear interface | mostly video understanding | predict latent → decode → robot-state extraction non-trivial |
| **Cosmos-Predict** (NVIDIA) | yes | text + previous-frame conditional, action conditioning is via Cosmos-1 + downstream model | video, manipulation showcased | heavy compute, frame-out not state-out |
| **DreamerV3** (DeepMind, OSS reimpls) | training script, not weights | yes (state + action → next latent) | yes if you train it on your task | needs per-task training, but the cleanest fit |
| **Genie 2** (DeepMind) | no public weights | yes | mostly games | not available |
| **1X World Model** (1X) | closed | unknown | yes | not available |

The honest read: **none of these is a drop-in.** Closing the gap
between what each model exposes and what `WorldModelBackend` expects
is real work. The current wedge is to make the slot exist and the
contract hold; integrations land one at a time, with eyes open about
what each one can and can't claim.

## Where the contract gets weird for learned substrates

Three things will need careful handling when a real world model lands:

1. **Provenance.** A world-model substrate has its own `model_sha256`
   that needs to land in the eval JSON next to the policy's. Two
   evals against the same policy on different world-model checkpoints
   are not comparable; the contract should say so.
2. **Success criterion.** Most success criteria today read positions
   off ground-truth physics (`object.lifted_mm > 50`). A world model
   doesn't have ground truth — it has its *prediction* of the
   position. The eval becomes "did the world model think the cube
   lifted," which is a different claim. Tasks will need a
   `success_kind: "predicted"` flag and probably side-by-side
   classical-sim grounding.
3. **Action repeat.** Classical sim runs at 200 Hz; learned models
   typically generate frames at 4–30 Hz. The `--action-repeat`
   contract invariant translates differently. Probably needs a
   per-backend default and a clear warning when the user crosses the
   line.

These are open problems, not blockers. Calling them out here so the
eval contract evolves with eyes open instead of accumulating silent
incompatibilities.

## What's not in scope

We are not going to build a world model. That's a billion-dollar bet
from labs with hundreds of researchers and we'd lose. The product is
the layer above. When World Labs / AMI / Cosmos ship something the
field wants to evaluate, RoboSandbox should be the obvious place to
score it.
