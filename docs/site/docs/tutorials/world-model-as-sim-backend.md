# Tutorial — world model as a sim backend

This page walks the `WorldModelBackend` slot end to end. It exists for
two reasons:

1. To prove that `robo-sandbox eval --sim-backend world_model` runs
   against a learned-dynamics propagator the same way it runs against
   MuJoCo or Newton — same CLI, same JSON schema, same Wilson CI,
   same provenance block.
2. To document, with code in front of you, what it would take to drop
   in V-JEPA-2, NVIDIA Cosmos, or your own DreamerV3 reimplementation.

For the framing — why we built this slot, where it sits relative to
the world-model wave, and what the eval contract has to learn to
handle learned substrates — see
[Eval for world models](../concepts/eval-for-world-models.md).

!!! warning "Status: slot, not a real world model"
    The reference predictor shipped with `WorldModelBackend` is
    `IdentityPredictor`. It returns `next_obs = current_obs` and
    therefore scores **0% on every task**. That is the point.
    Real world-model integrations replace `IdentityPredictor` with
    something that actually predicts dynamics; this page tells you
    where each named candidate would slot in and what the integration
    cost looks like.

## The slot

`WorldModelBackend` is a `SimBackend` like any other. It bootstraps
from a real backend (MuJoCo by default) for the initial observation
and scene metadata, then time-evolves purely via a
`WorldModelPredictor`:

```python
from robosandbox.protocols import SimBackend  # noqa: F401

class WorldModelPredictor(Protocol):
    def predict_next(
        self,
        obs: Observation,
        target_joints: np.ndarray | None,
        gripper: float | None,
    ) -> Observation: ...

    def reset(self) -> None: ...
```

A predictor returns a full `Observation`. Whatever fields it leaves
unchanged are carried over — that lets a joint-state-only model avoid
faking ee_pose or images it doesn't predict. Implementations should
be deterministic given `(obs, action)`; per-trial reset is the
predictor's responsibility.

## Smoke-testing the slot

```bash
robo-sandbox eval \
    --task pick_cube_franka \
    --policy runs/<some-recorded-episode> \
    --sim-backend world_model \
    --n-trials 4 --max-steps 200 \
    --output outputs/world_model_identity.json
```

Expected result: 0/4. The `IdentityPredictor` never moves the arm or
the cube, so no policy succeeds. The interesting part is the JSON:

```jsonc
{
  "schema_version": 2,
  "task": "pick_cube_franka",
  "sim_backend": "world_model",
  "n_trials": 4,
  "successes": 0,
  "rate": 0.0,
  "ci_low": 0.0, "ci_high": 0.602,
  "spatial_breakdown": { ... },
  "provenance": { ... }
}
```

That's the contract holding under a learned substrate. Wilson CI is
still defined at zero. `spatial_breakdown` still bins by initial cube
position. Provenance still has every field. The only thing missing —
and the next contract evolution this slot demands — is a
`world_model_sha256` field next to the policy's checkpoint hash, so
two evals against different world models are visibly non-comparable.
That's tracked in the [eval contract](../concepts/the-eval-contract.md)
spec.

## What plugging in a real model looks like

Each of the named candidates has a different integration shape. None
of them is a drop-in. The honest cost estimates below come from
reading their public model cards / reference implementations, not
from running them in this slot — that's the next round of work.

### V-JEPA-2 (Meta, weights public)

V-JEPA-2 predicts in latent space, not pixel or state space. To wrap
it as a `WorldModelPredictor` you need:

- An encoder run on the bootstrap observation (RGB → latent) at
  `predict_next` entry.
- The action embedded into the latent transition.
- A decoder back from latent → robot state (joint positions, ee_pose,
  scene_objects). This is the hard part — V-JEPA-2 doesn't ship a
  state decoder. You'd train one.

Realistic effort: weeks of training a state decoder before you can
even score a single trial. Worth it as a research project, not as a
weekend integration.

### NVIDIA Cosmos-Predict (weights public)

Cosmos predicts video frames conditional on text + previous frames.
Action conditioning is via Cosmos-1 + a downstream model. To wrap:

- Encode the bootstrap RGB.
- Generate the next frame conditioned on the action.
- Run a separate pose estimator on the predicted frame to recover
  joint positions and object poses.

Realistic effort: heavy compute (Cosmos models are large), and the
"pose estimator on a predicted frame" step is the same problem the
real-arm sim-to-real handoff faces — see
[Sim-to-real handoff](sim-to-real-handoff.md). The integration cost
is more about the GPU budget than the code.

### DreamerV3-style (training-script-public, weights per-task)

DreamerV3 is the cleanest fit on paper: it is exactly a `(state,
action) → next state` learned dynamics model. You'd:

- Train DreamerV3 on recorded trajectories from MuJoCo for your task.
- Wrap the trained world model's `imagine` step as `predict_next`.

Realistic effort: a few days of training per task, then the wrapper
is straightforward. The catch is that you have to train it per task
— it won't generalize across tasks the way V-JEPA / Cosmos try to.

### Genie 2 / 1X World Model

No public weights. Not integratable today.

## The trivial state-MLP option (if you want a real number above zero)

If you want a working number above 0% without integrating a real
world model, the smallest meaningful predictor is a state-only MLP
fit on recorded demos: `(joint_state_t, action_t) → joint_state_t+1`.
Train on a few hundred recorded trajectories; the network is small
enough to fit on a laptop CPU in minutes. Plug into
`WorldModelBackend` via the same `WorldModelPredictor` Protocol.

This isn't a world model in any serious sense — there's no scene
prediction, no perception coupling, no contact reasoning. But it is
a learned dynamics model and it scores >0% on tasks where the
ground-truth physics is "smooth enough" for a small MLP to
extrapolate. It exists as the natural next step beyond
`IdentityPredictor` if you want to verify the slot's downstream
behaviour (provenance, action repeat, success latching) end to end
without a billion-parameter dependency.

The implementation isn't shipped — it would take ~50 lines and one
short training script — and we'd rather you ship a real model than
an MLP. Treat it as a known fallback if your real-model integration
is blocked on weights or compute.

## Where this leaves us

The slot is real. `robo-sandbox eval --sim-backend world_model` runs.
The contract holds. Two things are explicitly *not* claimed by this
tutorial:

- We have not run any of the named real models in this slot. The
  table above is a planning document, not a results table.
- The eval contract does not yet have a `world_model_sha256`
  provenance field. Two evals against different world models would
  silently look comparable in the JSON — they aren't. This is the
  first concrete contract change that the wave forces, and it's
  tracked.

When a public model lands that's worth integrating, this is where
the integration goes.
