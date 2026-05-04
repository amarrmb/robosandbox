# Eval for World Models

The world-model wave is real. World Labs raised $230M and then $1B.
Yann LeCun's AMI Labs raised $1.03B at a $3.5B valuation. NVIDIA
Cosmos has open checkpoints. Meta's V-JEPA-2 weights are out. The
underlying bet across these labs is that learned dynamics will replace
classical physics simulators for robot training and evaluation.

If that bet pays off — even partially, even just for navigation
before manipulation — then "evaluate a policy in MuJoCo or Newton"
gets a third option: evaluate it in a learned world model. This page
is about what RoboSandbox does in that world.

## The Substrate Question Is Not the Eval Question

The substrate question is: what produces the next observation given
an action? Classical sim, GPU sim, a learned world model, or the
real arm. The eval contract question is: given a checkpoint and a
substrate, what counts as success, what's the schema for the result,
and how do you know two numbers are comparable?

World-model labs are competing on substrate. Nobody is currently
building eval contracts for their output. Today every world-model
paper hand-rolls its own success criteria, its own statistics, and
its own provenance. When the field has to compare two world models,
or compare two policies trained on two world models, somebody has to
ship the contract layer.

That's the layer this project occupies. RoboSandbox is not competing
with World Labs or AMI; the right read is that we should be sitting
on top of their substrates when those substrates are ready.

## World Model as a `SimBackend`

The `SimBackend` Protocol is small enough to fit in a paragraph: a
backend implements `load(scene)`, `reset()`, `step(target_joints,
gripper)`, `observe()`, `get_object_pose(id)`, `set_object_pose(id,
pose)`, `n_dof`, `joint_names`, and `close()`. Anything that can
answer *"given a state and an action, what's the next state"* fits.
Classical solvers fit. Learned dynamics models fit too — they just
produce the next observation differently.

The `experimental/newton-eval` branch ships a `WorldModelBackend`
that takes a `WorldModelPredictor` and runs it through the same
`robo-sandbox eval` pipeline as MuJoCo or Newton. The reference
predictor is intentionally trivial: it returns `next_obs =
current_obs`, scores 0% on every task, and exists to prove the slot
is wired without a real world-model dependency. The
[World model as a sim backend tutorial](../tutorials/world-model-as-sim-backend.md)
walks the wiring and the smoke-test result.

## What Plugging In a Real World Model Looks Like

We have not run any of the named real models in this slot. The
section below is a planning document, not a results table. Each
candidate has a different integration shape and a different cost.

V-JEPA-2 predicts in latent space, not pixel or state space. Wrapping
it as a `WorldModelPredictor` would require an encoder run on the
bootstrap observation at `predict_next` entry, the action embedded
into the latent transition, and a decoder back from latent to robot
state — which V-JEPA-2 doesn't ship. Training that decoder is its
own multi-week project before any eval trial runs. We do not have
this wired today.

NVIDIA Cosmos-Predict generates the next video frame conditional on
text and previous frames. Wrapping it would require encoding the
bootstrap RGB, generating the next frame on a real action, and
running a separate pose estimator on the predicted frame to recover
joint positions and object poses. The compute budget is the larger
problem; the integration code is the smaller one. We do not have
this wired today either.

DreamerV3-style learned dynamics is the cleanest fit on paper: it is
already a `(state, action) → next state` model. The catch is that
it has to be trained per task, on recorded trajectories from the
ground-truth simulator. A few days of training plus a thin wrapper
around the trained model's `imagine` step would land it. We have not
done this work yet, but the path is shorter than V-JEPA-2 or Cosmos.

Genie 2 has no public weights. The 1X World Model is closed. Neither
is integratable today.

## Where the Contract Gets Weird for Learned Substrates

Three things will need careful handling when a real world model
lands. None of them is a blocker, but each is a contract evolution
that should happen before two evals against different world models
silently look comparable in the JSON.

The first is provenance. A world-model substrate has its own
`model_sha256` that needs to land next to the policy's. The schema
doesn't carry this field today; bumping `schema_version` to 3 is
straightforward and is on the list once a real model is wired.

The second is the success criterion. Most success criteria today
read positions off ground-truth physics — for example,
`object.lifted_mm > 50`. A world model doesn't have ground truth; it
has its prediction of the position. The eval becomes "did the world
model think the cube lifted," which is a different claim and
probably needs a `success_kind: "predicted"` flag plus side-by-side
classical-sim grounding to be credible.

The third is action repeat. Classical sim runs at 200 Hz. Learned
models typically generate frames at 4–30 Hz. The `--action-repeat`
contract invariant translates differently between the two, and
needs a per-backend default plus a clear warning when the user
crosses the line.

## What's Not in Scope

We are not going to build a world model. That is a billion-dollar
bet from labs with hundreds of researchers, and we would lose. The
product is the layer above. When World Labs, AMI, or Cosmos ship
something the field wants to evaluate, RoboSandbox should be the
obvious place to score it.
