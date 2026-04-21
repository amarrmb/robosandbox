Output "/home/amar/robosandbox/docs/site/docs/assets/demos/lerobot_export.gif"

Set FontSize 14
Set Width 1200
Set Height 620
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set PlaybackSpeed 1.5
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "export MUJOCO_GL=osmesa"
Enter
Type "clear"
Enter
Show

Type "# A recorded episode already lives under runs/"
Sleep 700ms
Enter
Sleep 300ms
Type "ls runs/20260421-092948-ff946df8/"
Enter
Sleep 2000ms
Type ""
Enter

Type "# Export it to LeRobot v3 layout"
Sleep 700ms
Enter
Type "uv run robo-sandbox export-lerobot runs/20260421-092948-ff946df8 datasets/pick_demo"
Sleep 400ms
Enter
Sleep 4000ms
Type ""
Enter

Type "# Inspect what got written"
Sleep 700ms
Enter
Type "uv run python examples/inspect_lerobot_dataset.py datasets/pick_demo"
Sleep 400ms
Enter
Sleep 4000ms
Sleep 2000ms
