#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_deep_dive_graduate.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/.tmp/so101_handoff_v2.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/so101_handoff.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=6:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='Outgrow It on Purpose':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='A good starting point should make the next step easier.':fontcolor=0xcbd5e1:fontsize=23:x=(w-text_w)/2:y=290,\
drawtext=fontfile=${FONT}:text='LeRobot, MuJoCo, Isaac Sim, or real hardware can come later.':fontcolor=0xcbd5e1:fontsize=21:x=(w-text_w)/2:y=328,\
drawtext=fontfile=${FONT}:text='Phase 6: graduate when the problem demands a heavier stack':fontcolor=0x94a3b8:fontsize=20:x=(w-text_w)/2:y=382,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=0.7:duration=6.5,setpts=PTS-STARTPTS,\
drawbox=x=28:y=548:w=1224:h=140:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Keep the same interface while your needs grow.':fontcolor=white:fontsize=31:x=60:y=578,\
drawtext=fontfile=${FONT}:text='This is where you start moving from sim work toward real hardware.':fontcolor=0xcbd5e1:fontsize=20:x=60:y=626,\
setsar=1,format=yuv420p[handoff_term];\
[2:v]fps=30,trim=start=1.0:duration=8,setpts=PTS-STARTPTS,scale=1280:512:force_original_aspect_ratio=decrease,pad=1280:720:0:0:0x17172a,\
drawbox=x=28:y=548:w=1224:h=140:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='The point is not to stay here forever.':fontcolor=white:fontsize=31:x=60:y=578,\
drawtext=fontfile=${FONT}:text='Start here, then move to MuJoCo, Isaac Sim, LeRobot, or real hardware when needed.':fontcolor=0xcbd5e1:fontsize=20:x=60:y=626,\
setsar=1,format=yuv420p[handoff];\
[3:v]drawtext=fontfile=${FONT}:text='Phase 6: Graduate when needed':fontcolor=white:fontsize=44:x=(w-text_w)/2:y=205,\
drawtext=fontfile=${FONT}:text='Start here, then move on deliberately when the problem gets bigger.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=305,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=470,\
setsar=1,format=yuv420p[outro];\
[title][handoff_term][handoff][outro]concat=n=4:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
