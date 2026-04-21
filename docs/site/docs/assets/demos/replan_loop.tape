Output "/home/amar/robosandbox/docs/site/docs/assets/demos/replan_loop.gif"

Set FontSize 14
Set Width 1100
Set Height 560
Set Theme "Catppuccin Mocha"
Set TypingSpeed 32ms
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "export MUJOCO_GL=osmesa"
Enter
Type "clear"
Enter
Show

Type "# Task: a red cube placed 1.2m out (Franka's reach is ~0.85m)"
Sleep 900ms
Enter
Sleep 300ms
Type "uv run --quiet python examples/replan_demo.py"
Sleep 500ms
Enter
Wait
Sleep 2000ms
