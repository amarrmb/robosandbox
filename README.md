# RoboSandbox

> Sim-first agentic manipulation sandbox.
> Any arm. Any object. Any command.

```
┌─────────────────────────────────────────────────────────┐
│  user types:  "pick the blue cube, put it on the plate" │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
            VLM plan → skills → IK → sim → result
                          │
                          ▼
                record MCAP → (opt-in) DeviceNexus
                          │
                          ▼
           fine-tune policy → hot-load back as a skill
```

## Status

v0.1 is under active construction. The hello-pick vertical slice runs:

```bash
git clone <this-repo> robosandbox
cd robosandbox
uv sync
uv run python -m robosandbox.demo
# -> runs/<timestamp>/video.mp4
```

## Layout

```
packages/
├── robosandbox-core/               # Apache-2.0, the only required install
├── robosandbox-curobo/             # (planned) GPU motion planning
├── robosandbox-molmo/              # (planned) dedicated pointing model
├── robosandbox-anygrasp/           # (planned) research-license grasping
└── robosandbox-dn/                 # (planned) DeviceNexus recorder + learned skills
```

## License

Core: Apache 2.0. Optional `contrib/` plugins carry their own licenses.
