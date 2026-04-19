Output "/home/amar/robosandbox/docs/site/docs/assets/demos/lerobot_export.gif"

Set FontSize 14
Set Width 1200
Set Height 780
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "clear"
Enter
Show

Type "# 1. Record a pick episode (writes video.mp4 + events.jsonl + result.json)"
Sleep 900ms
Enter
Sleep 300ms
Type "uv run python examples/record_demo.py --out-dir runs 2>/dev/null"
Sleep 400ms
Enter
Sleep 10000ms

Type "RUN=$(ls -1t runs/ | head -1); echo 'Latest run: '$RUN"
Enter
Sleep 1500ms
Type ""
Enter

Type "# 2. Export that run to LeRobot v3 layout"
Sleep 700ms
Enter
Type "uv run robo-sandbox export-lerobot runs/$RUN datasets/pick_demo"
Sleep 400ms
Enter
Sleep 2500ms
Type ""
Enter

Type "# 3. Inspect what got written"
Sleep 700ms
Enter
Type "uv run python examples/inspect_lerobot_dataset.py datasets/pick_demo"
Sleep 400ms
Enter
Sleep 3500ms
Sleep 1500ms
