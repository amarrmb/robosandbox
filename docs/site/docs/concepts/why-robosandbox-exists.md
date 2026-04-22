# Why RoboSandbox Exists

If you are new to robotics, the jargon gets in the way fast.

You hear names like `ACT`, `smolVLA`, `pi0.7`, `GROOT`, or "world
model" and it starts to sound like every system is solving the same
problem.

They are not.

The easiest way to stay oriented is to start with the robot loop, not
the model names:

```text
task
-> observations
-> decision
-> action
-> outcome
```

This memo is about that loop and why RoboSandbox exists around it.

## The real problem

Most robotics teams do not fail because they lack one more model.

They fail because turning an idea into a testable robot behavior takes
too much work. A simple question like:

- can this arm reach the object?
- can this camera see the pre-grasp state?
- will this policy recover if the first attempt fails?

quickly turns into glue code, mismatched interfaces, weak evaluation,
and expensive real-world tests.

The deeper problem is that the layers of the system get mixed together.
People talk about the task, the model, the robot, the camera, and the
evaluation loop as if they are the same thing. They are not.

## Start with the task

A **task** is the job you want the robot to do.

Examples:

- pick the red cube
- place the tomato in the bin
- open the drawer
- move the seedling into the tray cell

That sounds simple, but a task is never just the sentence.

Take:

```text
"harvest the ripe tomato"
```

Now the real questions show up:

- where is the tomato?
- how do we know it is ripe?
- can the camera see it?
- how should the robot approach it?
- what counts as success?

So in practice, a task is:

```text
goal
+ observations needed
+ action structure
+ success test
```

That is why "expressing the task" matters. It means turning a vague idea
into something the system can actually run and evaluate.

This is not just about clarity. It fixes the evaluation surface:

- what observations are required
- what action space is assumed
- what success metric is scored
- what constraints and safety checks matter

Without that, teams end up building the wrong thing:

- the wrong camera setup
- the wrong motion primitive
- the wrong recovery logic
- the wrong success metric

## Skills are smaller than tasks

A **skill** is a reusable robot action.

Examples:

- `pick`
- `place_on`
- `push`
- `home`
- `open_drawer`

If the task is the job, the skill is one action inside that job.

```text
Task:
"pick the red cube and put it on the green cube"

Possible skills:
- pick(red cube)
- place_on(green cube)
```

This distinction matters because you want to change tasks without having
to rewrite the robot's whole action vocabulary.

## Policies and world models solve different problems

A **policy** is the part that looks at the current observation and
chooses what to do next.

In plain language:

```text
what the robot sees now
-> what command it should send next
```

Many modern robotics models live mostly in this layer.

Models like `ACT`, `smolVLA`, `pi0.7`, and `GROOT` are useful to think
of as primarily action-selection systems:

```text
image + robot state
-> model
-> next action
```

That is a simplification, not a taxonomy. Some of these model families
span multiple layers of the stack. They are placed here by their
primary role in this memo.

What matters is the boundary:

a strong policy still does not answer:

- what task are we solving?
- what counts as success?
- what camera setup does it assume?
- what robot embodiment does it assume?

A **world model** is different. It is closer to:

```text
current state + action
-> predicted future state
```

or:

```text
current observation
-> internal model of how the world evolves
```

That prediction can help with:

- planning
- action scoring
- imagining rollouts
- learning a compact state of the world

So if someone says "we use a world model for robotics", the next
question is: what job is it doing in the stack? Is it choosing actions
directly? Helping a planner? Acting like a learned simulator? Those are
different claims.

## Workflow is where time gets burned

A **workflow** is the full loop:

```text
define task
-> run planner or policy
-> execute actions
-> record results
-> inspect failure
-> try again
```

This is where teams often lose the most time.

Not because one model is impossible, but because the whole loop is hard
to run, hard to compare, and hard to debug.

That is why these distinctions matter:

```text
task = what job are we trying to do?
skill = what reusable action do we have?
policy = how do we choose the next action?
world model = what do we think will happen next?
workflow = how do we run, record, inspect, and improve the system?
```

Without those boundaries, teams blame the wrong part of the system:

- bad task definition blamed on the policy
- embodiment mismatch blamed on the model
- bad camera placement blamed on sim
- weak evaluation workflow blamed on the dataset

## What this means in practice

```text
task
-> observation
-> decision
-> action
-> outcome
```

If you want to work on robotics sanely, that visibility needs to be
concrete, not philosophical. In practice it means the system should
give you things you can inspect:

- the task you ran
- the observations the robot saw
- the actions or skills chosen
- the commands sent
- the result and failure reason
- a replayable record of what happened

That is the real meaning of an **inspectable experimentation layer**.

It also gives you a stable evaluation loop for comparing approaches
against the same contract:

- same task definition
- same observation schema
- same action schema
- same recording and evaluation loop

That does **not** mean identical transfer across different robots,
cameras, or deployment environments. It means you can compare methods
without changing the whole experimental surface each time.

## Where RoboSandbox fits

RoboSandbox exists for a narrow reason:

to reduce the cost of trying robotics ideas and make failures easier to
localize.

It is not:

- the policy
- the world model
- the perfect simulator
- the final production robotics stack

It is the layer where you define the task clearly, run the workflow,
record what happened, and compare approaches against the same
evaluation contract.

That matters when you want to compare:

- a hand-written baseline
- an `ACT`-style policy
- a `smolVLA`-style model
- a `pi0.7` or `GROOT`-like action model
- a world-model-assisted planner

Use RoboSandbox when your bottleneck is:

- defining tasks clearly
- wiring experiments too slowly
- comparing approaches inconsistently
- reproducing failures poorly
- moving from idea to first real-robot test too slowly

Do not use RoboSandbox when your bottleneck is:

- maximum physics fidelity
- photorealistic rendering
- large-scale synthetic data generation
- production deployment infrastructure

If RoboSandbox does not lower:

- engineering time per experiment
- glue code between components
- time to reproduce failures
- time to first real-world validation

then it is not doing its job.
