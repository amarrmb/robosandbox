Output "/home/amar/robosandbox/docs/site/docs/assets/demos/so100_policy.gif"

Set FontSize 14
Set Width 1200
Set Height 540
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set PlaybackSpeed 2.0
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "export MUJOCO_GL=osmesa"
Enter
Type "clear"
Enter
Show

Type "# Non-bundled SO-ARM100 + public LeRobot ACT checkpoint, end to end"
Sleep 900ms
Enter
Sleep 300ms
Type "uv run --quiet python examples/so_arm100/run_so100_policy.py"
Sleep 400ms
Enter
Sleep 120000ms
Sleep 2000ms
