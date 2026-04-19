Output "/home/amar/robosandbox/docs/site/docs/assets/demos/ycb_mug_bench.gif"

Set FontSize 16
Set Width 1100
Set Height 520
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set Padding 20

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "clear"
Enter
Show

Type "# Mesh-import acceptance test: Franka picks the bundled YCB mug"
Sleep 1200ms
Enter
Sleep 500ms
Type "uv run --quiet robo-sandbox-bench --tasks pick_ycb_mug --seeds 1"
Sleep 700ms
Enter
Sleep 7000ms
Sleep 2000ms
