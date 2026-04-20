#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_deep_dive_run_inspect.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/franka_pick.gif" \
  -ignore_loop 0 -i "$ROOT/.tmp/record_artifacts_v2.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/run_artifacts.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=6:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='Run It, Then Inspect It':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='The point is not just to watch motion. The point is to see what the run produced and debug from there.':fontcolor=0xcbd5e1:fontsize=20:x=(w-text_w)/2:y=295,\
drawtext=fontfile=${FONT}:text='Phase 4: make runs inspectable enough to trust your next change':fontcolor=0x94a3b8:fontsize=20:x=(w-text_w)/2:y=355,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=1:duration=7,setpts=PTS-STARTPTS,scale=1280:960:force_original_aspect_ratio=increase,crop=1280:720,\
drawbox=x=28:y=558:w=1224:h=130:color=0x0f172a@0.86:t=fill,\
drawtext=fontfile=${FONT}:text='First run the task end to end.':fontcolor=white:fontsize=34:x=60:y=588,\
drawtext=fontfile=${FONT}:text='A real loop gives you something concrete to reason about.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=636,\
setsar=1,format=yuv420p[run];\
[2:v]fps=30,trim=start=0.8:duration=6.2,setpts=PTS-STARTPTS,\
drawbox=x=28:y=548:w=1224:h=140:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Then inspect the artifacts.':fontcolor=white:fontsize=34:x=60:y=578,\
drawtext=fontfile=${FONT}:text='video.mp4, events.jsonl, and result.json let you debug the outcome.':fontcolor=0xcbd5e1:fontsize=21:x=60:y=626,\
setsar=1,format=yuv420p[artifacts];\
[3:v]fps=30,trim=start=3.0:duration=8,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:0x17172a,\
drawbox=x=28:y=548:w=1224:h=140:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Inspection is part of the workflow, not an afterthought.':fontcolor=white:fontsize=31:x=60:y=578,\
drawtext=fontfile=${FONT}:text='See what the run produced before you change the robot, task, or policy.':fontcolor=0xcbd5e1:fontsize=20:x=60:y=626,\
setsar=1,format=yuv420p[inspect];\
[4:v]drawtext=fontfile=${FONT}:text='Phase 4: Run and inspect':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='If you cannot inspect the run, you are mostly guessing.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=305,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=470,\
setsar=1,format=yuv420p[outro];\
[title][run][artifacts][inspect][outro]concat=n=5:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
