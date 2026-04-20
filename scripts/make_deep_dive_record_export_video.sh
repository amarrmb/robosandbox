#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_deep_dive_record_export.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/franka_pick.gif" \
  -ignore_loop 0 -i "$ROOT/.tmp/record_artifacts_v2.gif" \
  -ignore_loop 0 -i "$ROOT/.tmp/lerobot_export_v2.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=6:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='Record Once, Reuse the Result':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='A single run can become artifacts for debugging and data for downstream tools.':fontcolor=0xcbd5e1:fontsize=22:x=(w-text_w)/2:y=295,\
drawtext=fontfile=${FONT}:text='Phase 5: collect data without rebuilding the workflow':fontcolor=0x94a3b8:fontsize=21:x=(w-text_w)/2:y=360,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=0.8:duration=7.5,setpts=PTS-STARTPTS,scale=1280:960:force_original_aspect_ratio=increase,crop=1280:720,\
drawbox=x=28:y=558:w=1224:h=130:color=0x0f172a@0.86:t=fill,\
drawtext=fontfile=${FONT}:text='Start with a real recorded run':fontcolor=white:fontsize=34:x=60:y=588,\
drawtext=fontfile=${FONT}:text='The same loop the project uses can also leave usable data behind.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=636,\
setsar=1,format=yuv420p[run];\
[2:v]fps=30,trim=start=0.7:duration=6.5,setpts=PTS-STARTPTS,\
drawbox=x=28:y=548:w=1224:h=140:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Step 1: inspect the artifacts':fontcolor=white:fontsize=34:x=60:y=578,\
drawtext=fontfile=${FONT}:text='video.mp4, events.jsonl, and result.json tell you what happened.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=626,\
setsar=1,format=yuv420p[inspect];\
[3:v]fps=30,trim=start=1.6:duration=10.5,setpts=PTS-STARTPTS,\
drawbox=x=28:y=548:w=1224:h=140:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Step 2: export the same run':fontcolor=white:fontsize=34:x=60:y=578,\
drawtext=fontfile=${FONT}:text='Use export-lerobot, then inspect the dataset with normal tools.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=626,\
setsar=1,format=yuv420p[export];\
[4:v]drawtext=fontfile=${FONT}:text='Why this matters':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=185,\
drawtext=fontfile=${FONT}:text='You are not recording data in a separate universe.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=280,\
drawtext=fontfile=${FONT}:text='The same run you inspect can also become a portable dataset.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=320,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=470,\
setsar=1,format=yuv420p[outro];\
[title][run][inspect][export][outro]concat=n=5:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
