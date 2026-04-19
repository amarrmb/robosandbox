Output "/home/amar/robosandbox/docs/site/docs/assets/demos/reachability.gif"

Set FontSize 14
Set Width 1200
Set Height 600
Set Theme "Catppuccin Mocha"
Set TypingSpeed 32ms
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "clear"
Enter
Show

Type "# Authoring-time check: catch bad object placements before physics runs"
Sleep 900ms
Enter
Sleep 300ms
Type "uv run --quiet python examples/run_custom_task.py examples/tasks/pick_unreachable_cube.yaml"
Sleep 500ms
Enter
Sleep 6500ms
Sleep 1500ms
