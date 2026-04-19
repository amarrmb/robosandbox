Output "/home/amar/robosandbox/docs/site/docs/assets/demos/custom_task.gif"

Set FontSize 15
Set Width 1100
Set Height 540
Set Theme "Catppuccin Mocha"
Set TypingSpeed 32ms
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "clear"
Enter
Show

Type "# Peek at the user-authored YAML"
Sleep 800ms
Enter
Sleep 300ms
Type "cat examples/tasks/pick_yellow_cube.yaml | head -18"
Sleep 500ms
Enter
Sleep 3500ms
Type ""
Enter
Type "# Run it through the agent loop across 3 randomized seeds"
Sleep 700ms
Enter
Type "uv run --quiet python examples/run_custom_task.py examples/tasks/pick_yellow_cube.yaml --seeds 3"
Sleep 500ms
Enter
Sleep 6500ms
Sleep 1500ms
