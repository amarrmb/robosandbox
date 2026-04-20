#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_deep_dive_choose_robot.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/franka_pick.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/so100_policy.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/so100_policy_run.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=6:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='Pick the Embodiment First':fontcolor=white:fontsize=50:x=(w-text_w)/2:y=195,\
drawtext=fontfile=${FONT}:text='Start with a bundled robot, then swap in your own embodiment when the question changes.':fontcolor=0xcbd5e1:fontsize=22:x=(w-text_w)/2:y=285,\
drawtext=fontfile=${FONT}:text='Phase 1: learn the loop before you optimize the robot choice':fontcolor=0x94a3b8:fontsize=21:x=(w-text_w)/2:y=350,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=1:duration=7,setpts=PTS-STARTPTS,scale=1280:960:force_original_aspect_ratio=increase,crop=1280:720,\
drawbox=x=28:y=558:w=1224:h=130:color=0x0f172a@0.86:t=fill,\
drawtext=fontfile=${FONT}:text='Bundled robots get you moving immediately.':fontcolor=white:fontsize=34:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Use the built-in arm or the bundled Franka to learn the loop.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=636,\
setsar=1,format=yuv420p[bundled];\
[2:v]fps=30,trim=start=1.0:duration=8,setpts=PTS-STARTPTS,scale=1280:576:force_original_aspect_ratio=decrease,pad=1280:720:0:0:0x17172a,setsar=1,format=yuv420p[term];\
[3:v]fps=30,trim=duration=2.6,setpts=PTS-STARTPTS,scale=360:270,setsar=1,format=yuv420p[simsmall];\
[term][simsmall]overlay=x=888:y=336,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Non-bundled robots use the same workflow.':fontcolor=white:fontsize=33:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Bring your own URDF or MJCF, then keep the rest of the stack.':fontcolor=0xcbd5e1:fontsize=21:x=60:y=636,\
drawbox=x=876:y=324:w=388:h=294:color=0x0f172a@0.90:t=fill,\
drawtext=fontfile=${FONT}:text='non-bundled embodiment':fontcolor=0xcbd5e1:fontsize=18:x=928:y=596,\
setsar=1,format=yuv420p[nonbundled];\
[4:v]drawtext=fontfile=${FONT}:text='Phase 1: Choose a robot':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=200,\
drawtext=fontfile=${FONT}:text='Use the easiest embodiment that still answers your current question.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=305,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=470,\
setsar=1,format=yuv420p[outro];\
[title][bundled][nonbundled][outro]concat=n=4:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
