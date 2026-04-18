# robosandbox

Sim-first agentic manipulation sandbox: any arm, any object, any command.

This is the core package. Install it:

```bash
pip install robosandbox
```

Run the hello-pick demo:

```bash
python -m robosandbox.demo
# writes runs/<timestamp>-<id>/video.mp4 + events.jsonl + result.json
```

No VLM, no API keys, no GPU required for the demo.

See the repo README for the full architecture, plugin catalog, and
roadmap.
