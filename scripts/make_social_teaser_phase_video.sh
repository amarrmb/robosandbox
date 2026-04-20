#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_teaser_phase.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/franka_pick.gif" \
  -ignore_loop 0 -i "$ROOT/.tmp/record_artifacts_v2.gif" \
  -ignore_loop 0 -i "$ROOT/.tmp/lerobot_export_v2.gif" \
  -ignore_loop 0 -i "$ROOT/.tmp/so101_handoff_v2.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=5:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='RoboSandbox':fontcolor=white:fontsize=68:x=(w-text_w)/2:y=110,\
drawtext=fontfile=${FONT}:text='A simpler starting point for robot manipulation work':fontcolor=white:fontsize=32:x=(w-text_w)/2:y=238,\
drawtext=fontfile=${FONT}:text='Pick an embodiment. Choose the source of action. Define the task.':fontcolor=0xcbd5e1:fontsize=23:x=(w-text_w)/2:y=302,\
drawtext=fontfile=${FONT}:text='Run the loop, inspect the result, and grow from there.':fontcolor=0xcbd5e1:fontsize=23:x=(w-text_w)/2:y=348,\
drawtext=fontfile=${FONT}:text='Good for learning the loop before you need bigger stacks.':fontcolor=0x94a3b8:fontsize=21:x=(w-text_w)/2:y=430,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=1:duration=10,setpts=PTS-STARTPTS,scale=1280:960:force_original_aspect_ratio=increase,crop=1280:720,\
drawbox=x=28:y=538:w=1224:h=150:color=0x0f172a@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='1. Choose a robot':fontcolor=white:fontsize=30:x=60:y=566,\
drawtext=fontfile=${FONT}:text='2. Choose how it acts':fontcolor=white:fontsize=30:x=60:y=606,\
drawtext=fontfile=${FONT}:text='3. Define the task and run the loop':fontcolor=white:fontsize=30:x=60:y=646,\
setsar=1,format=yuv420p[phases123];\
[2:v]fps=30,trim=start=1.0:duration=5.5,setpts=PTS-STARTPTS,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='4. Run and inspect':fontcolor=white:fontsize=34:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Every run leaves artifacts you can inspect and debug.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=636,\
setsar=1,format=yuv420p[phase4];\
[3:v]fps=30,trim=start=2.4:duration=6.5,setpts=PTS-STARTPTS,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='5. Collect data':fontcolor=white:fontsize=34:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Export the same run to LeRobot format and keep going.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=636,\
setsar=1,format=yuv420p[phase5];\
[4:v]fps=30,trim=start=0.8:duration=5.5,setpts=PTS-STARTPTS,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='6. Graduate when needed':fontcolor=white:fontsize=34:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Keep the workflow, then move to bigger stacks or real hardware.':fontcolor=0xcbd5e1:fontsize=22:x=60:y=636,\
setsar=1,format=yuv420p[phase6];\
[5:v]drawtext=fontfile=${FONT}:text='Start here. Grow out of it on purpose.':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=220,\
drawtext=fontfile=${FONT}:text='MuJoCo. Isaac Sim. LeRobot. Real hardware.':fontcolor=0xd1d5db:fontsize=28:x=(w-text_w)/2:y=330,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=480,\
setsar=1,format=yuv420p[outro];\
[title][phases123][phase4][phase5][phase6][outro]concat=n=6:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
