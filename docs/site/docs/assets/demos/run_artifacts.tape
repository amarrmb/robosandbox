Output "/home/amar/robosandbox/docs/site/docs/assets/demos/run_artifacts.gif"

Set FontSize 15
Set Width 1100
Set Height 680
Set Theme "Catppuccin Mocha"
Set TypingSpeed 30ms
Set Padding 18

Hide
Type "cd /home/amar/robosandbox"
Enter
Type "clear"
Enter
Show

Type "# Every successful run lands in runs/<stamp>/ with three artifacts"
Sleep 900ms
Enter
Sleep 300ms
Type "ls -lh runs/20260421-092948-ff946df8/"
Sleep 500ms
Enter
Sleep 2500ms
Type ""
Enter
Type "# result.json: one-line verdict"
Sleep 800ms
Enter
Type "cat runs/20260421-092948-ff946df8/result.json"
Sleep 500ms
Enter
Sleep 2000ms
Type ""
Enter
Type "# events.jsonl: per-frame robot + object state"
Sleep 900ms
Enter
Type "head -1 runs/20260421-092948-ff946df8/events.jsonl | python3 -m json.tool | head -16"
Sleep 500ms
Enter
Sleep 3000ms
Sleep 1500ms
