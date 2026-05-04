# Tutorial — World Model as a Sim Backend

This page walks the `WorldModelBackend` slot end to end. It exists
for two reasons. First, to prove that `robo-sandbox eval --sim-backend
world_model` runs against a learned-dynamics propagator the same way
it runs against MuJoCo or Newton — same CLI, same JSON schema, same
Wilson CI, same provenance block. Second, to document, with code in
front of you, what it would take to drop in V-JEPA-2, NVIDIA Cosmos,
or your own DreamerV3 reimplementation.

The framing for why this slot exists at all — where it sits relative
to the world-model wave, what the eval contract has to learn to
handle learned substrates — is in
[Eval for World Models](../concepts/eval-for-world-models.md).

The reference predictor shipped with `WorldModelBackend` is
`IdentityPredictor`. It returns `next_obs = current_obs` and scores
0% on every task. That is intentional. The wedge is the slot, not a
world-model implementation. Real world-model integrations replace
`IdentityPredictor` with something that actually predicts dynamics.

## The Slot

`WorldModelBackend` is a `SimBackend` like any other. It bootstraps
from a real backend (MuJoCo by default) for the initial observation
and scene metadata, then time-evolves purely via a
`WorldModelPredictor`:

```python
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
unchanged are carried over from the input — that lets a
joint-state-only model avoid faking the ee_pose or images it doesn't
predict. Implementations should be deterministic given `(obs,
action)`. Per-trial reset is the predictor's responsibility; the eval
CLI calls `policy.reset()` between trials and treats the backend as
black-box.

## Smoke-Testing the Slot

```bash
robo-sandbox eval \
    --task pick_cube_franka \
    --policy runs/<some-recorded-episode> \
    --sim-backend world_model \
    --n-trials 4 --max-steps 200 \
    --output outputs/world_model_identity.json
```

The expected result is 0/4. `IdentityPredictor` never moves the arm
or the cube, so no policy succeeds. The interesting part is the JSON:

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

That's the contract holding under a learned substrate. The Wilson CI
is still defined at zero. `spatial_breakdown` still bins by initial
cube position. Provenance still has every field. The only thing
missing — and the next contract evolution this slot demands — is a
`world_model_sha256` field next to the policy's checkpoint hash, so
two evals against different world models are visibly non-comparable.
That work is tracked in [the eval contract](../concepts/the-eval-contract.md).

## What Plugging In a Real Model Looks Like

We have not run any of the named real models in this slot. The cost
estimates below come from reading model cards and reference
implementations, not from hands-on integration. Each candidate has a
different shape and a different blocker.

V-JEPA-2 predicts in latent space, not pixel or state space. Wrapping
it requires running the encoder on the bootstrap observation at
`predict_next` entry, embedding the action into the latent
transition, and decoding back from latent to robot state. V-JEPA-2
doesn't ship a state decoder. You'd train one. Realistic effort:
weeks of training a state decoder before any eval trial runs. Worth
it as a research project, not as a weekend integration.

NVIDIA Cosmos-Predict generates the next video frame conditional on
text and previous frames. Action conditioning is via Cosmos-1 plus a
downstream model. Wrapping it requires encoding the bootstrap RGB,
generating the next frame on a real action, and running a separate
pose estimator on the predicted frame to recover joint positions and
object poses. The compute budget is the larger problem; the
integration code is the smaller one.

DreamerV3-style learned dynamics is the cleanest fit on paper. It is
already a `(state, action) → next state` model. You'd train DreamerV3
on recorded trajectories from MuJoCo for the task, then wrap the
trained world model's `imagine` step as `predict_next`. Effort: a
few days of training per task plus a thin wrapper. The catch is that
it has to be trained per task — it won't generalize across tasks the
way V-JEPA or Cosmos try to.

Genie 2 has no public weights. The 1X World Model is closed. Neither
is integratable today.

## A Trivial State-MLP Option

If you want a working number above zero without integrating a real
world model, the smallest meaningful predictor is a state-only MLP
fit on recorded demos: `(joint_state_t, action_t) → joint_state_t+1`.
Train on a few hundred recorded trajectories; the network is small
enough to fit on a laptop CPU in minutes. Plug it into
`WorldModelBackend` via the same `WorldModelPredictor` Protocol.

This isn't a world model in any serious sense. There's no scene
prediction, no perception coupling, no contact reasoning. But it is
a learned dynamics model, and it scores above 0% on tasks where the
ground-truth physics is smooth enough for a small MLP to extrapolate.
It exists as the natural next step beyond `IdentityPredictor` for
verifying the slot's downstream behaviour (provenance, action repeat,
success latching) end to end without a billion-parameter dependency.

The implementation isn't shipped. We'd rather you ship a real model
than an MLP, and the MLP would take about 50 lines and one short
training script if needed as a fallback.

## Where This Leaves Us

The slot is real. `robo-sandbox eval --sim-backend world_model` runs.
The contract holds. Two things this tutorial does not claim:

We have not run any of the named real models in this slot. The table
above is a planning document, not a results table.

The eval contract does not yet have a `world_model_sha256` provenance
field. Two evals against different world models would silently look
comparable in the JSON today. That is the first concrete contract
change the wave forces, and it is tracked in
[the eval contract](../concepts/the-eval-contract.md). When a public
model lands that's worth integrating, this is where the integration
goes.
