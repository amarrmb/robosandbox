Output "/home/amar/robosandbox/docs/site/docs/assets/demos/phases_log.gif"

Set FontSize 16
Set Width 1100
Set Height 420
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set PlaybackSpeed 2.5
Set Padding 20

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "export MUJOCO_GL=osmesa"
Enter
Type "clear"
Enter
Show

Type "# Watch the agent's PLAN → EXECUTE phases printed by the bench runner"
Sleep 1000ms
Enter
Sleep 400ms
Type "uv run --quiet robo-sandbox-bench --tasks pick_cube_franka --seeds 1"
Sleep 500ms
Enter
Sleep 50000ms
Sleep 2000ms
