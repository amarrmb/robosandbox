Output "/home/amar/robosandbox/docs/site/docs/assets/demos/so101_handoff.gif"

Set FontSize 14
Set Width 1200
Set Height 480
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "export MUJOCO_GL=osmesa"
Enter
Type "clear"
Enter
Show

Type "# A RealRobotBackend subclass runs the same Home skill the sim uses"
Sleep 900ms
Enter
Sleep 300ms
Type "uv run --quiet python examples/so101_handoff/run_home_skill.py"
Sleep 400ms
Enter
Wait
Sleep 2000ms
