#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/site/docs/assets/demos/robosandbox_deep_dive_choose_actuation.mp4}"
FONT="/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"

mkdir -p "$(dirname "$OUT")"

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1280x720:d=3:r=30" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/vlm_walkthrough.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/franka_pick.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/so100_policy.gif" \
  -ignore_loop 0 -i "$ROOT/docs/site/docs/assets/demos/so100_policy_run.gif" \
  -f lavfi -i "color=c=0x111827:s=1280x720:d=6:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='Choose the Source of Action':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=195,\
drawtext=fontfile=${FONT}:text='Start with planner skills for legibility, or plug in a policy when you need model-driven actions.':fontcolor=0xcbd5e1:fontsize=21:x=(w-text_w)/2:y=285,\
drawtext=fontfile=${FONT}:text='Phase 2: change the controller without changing the rest of the loop':fontcolor=0x94a3b8:fontsize=20:x=(w-text_w)/2:y=350,\
setsar=1,format=yuv420p[title];\
[1:v]fps=30,trim=start=1.2:duration=7.5,setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:0x17172a,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Planner mode: start from a task description.':fontcolor=white:fontsize=33:x=60:y=588,\
drawtext=fontfile=${FONT}:text='The planner turns task text into tool calls and skill steps.':fontcolor=0xcbd5e1:fontsize=21:x=60:y=636,\
setsar=1,format=yuv420p[planner_term];\
[2:v]fps=30,trim=start=1:duration=5.5,setpts=PTS-STARTPTS,scale=1280:960:force_original_aspect_ratio=increase,crop=1280:720,\
drawbox=x=28:y=558:w=1224:h=130:color=0x0f172a@0.86:t=fill,\
drawtext=fontfile=${FONT}:text='That plan still runs through the same loop.':fontcolor=white:fontsize=33:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Same environment, same artifacts, different source of action.':fontcolor=0xcbd5e1:fontsize=21:x=60:y=636,\
setsar=1,format=yuv420p[planner_run];\
[3:v]fps=30,trim=start=1.0:duration=8,setpts=PTS-STARTPTS,scale=1280:576:force_original_aspect_ratio=decrease,pad=1280:720:0:0:0x17172a,setsar=1,format=yuv420p[policyterm];\
[4:v]fps=30,trim=duration=2.6,setpts=PTS-STARTPTS,scale=360:270,setsar=1,format=yuv420p[policy_sim];\
[policyterm][policy_sim]overlay=x=888:y=336,\
drawbox=x=28:y=558:w=1224:h=130:color=0x111827@0.88:t=fill,\
drawtext=fontfile=${FONT}:text='Policy mode: swap in a checkpoint.':fontcolor=white:fontsize=33:x=60:y=588,\
drawtext=fontfile=${FONT}:text='Use the policy seam when you want actions to come from a model.':fontcolor=0xcbd5e1:fontsize=21:x=60:y=636,\
drawbox=x=876:y=324:w=388:h=294:color=0x0f172a@0.90:t=fill,\
drawtext=fontfile=${FONT}:text='policy replay in sim':fontcolor=0xcbd5e1:fontsize=18:x=950:y=596,\
setsar=1,format=yuv420p[policy];\
[5:v]drawtext=fontfile=${FONT}:text='Phase 2: Choose how it acts':fontcolor=white:fontsize=46:x=(w-text_w)/2:y=200,\
drawtext=fontfile=${FONT}:text='Swap planners and policies without losing inspectability.':fontcolor=0xd1d5db:fontsize=24:x=(w-text_w)/2:y=305,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=30:x=(w-text_w)/2:y=470,\
setsar=1,format=yuv420p[outro];\
[title][planner_term][planner_run][policy][outro]concat=n=5:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
