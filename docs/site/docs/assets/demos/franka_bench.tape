Output "/home/amar/robosandbox/docs/site/docs/assets/demos/franka_bench.gif"

Set FontSize 16
Set Width 1100
Set Height 520
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set PlaybackSpeed 1.0
Set Padding 20

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "clear"
Enter
Show

Type "# Run the acceptance test that proves the Franka URDF loads end-to-end"
Sleep 1200ms
Enter
Sleep 500ms
Type "uv run --quiet robo-sandbox-bench --tasks pick_cube_franka --seeds 1"
Sleep 700ms
Enter
Sleep 6500ms
Sleep 2000ms
