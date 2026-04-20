#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_deep_dive_define_task.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/custom_task.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=6:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='Define the Task, Not the Stack':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='Describe what success looks like, rerun the loop, and keep the surrounding workflow intact.':fontcolor=0xcbd5e1:fontsize=21:x=(w-text_w)/2:y=295,\
drawtext=fontfile=${FONT}:text='Phase 3: change the task contract instead of rewriting infrastructure':fontcolor=0x94a3b8:fontsize=20:x=(w-text_w)/2:y=355,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=2.5:duration=14,setpts=PTS-STARTPTS,scale=1280:628:force_original_aspect_ratio=decrease,pad=1280:720:0:0:0x17172a,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Tasks are the contract for what you want tested.':fontcolor=white:fontsize=33:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Write one, run it, and keep the rest of the loop the same.':fontcolor=0xcbd5e1:fontsize=21:x=60:y=636,\
setsar=1,format=yuv420p[task];\
[2:v]drawtext=fontfile=${FONT}:text='Phase 3: Define the task':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='Change the question you are asking, not the whole stack around it.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=305,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=470,\
setsar=1,format=yuv420p[outro];\
[title][task][outro]concat=n=3:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
