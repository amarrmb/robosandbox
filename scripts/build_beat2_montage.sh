#!/usr/bin/env bash
# Beat 2 montage: 256-world wall + 4-world zoom + title cards.
# Run after rendering reach_256w_v3.mp4 and reach_4w_v3.mp4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs/beat2/beat2_reach_montage.mp4"
WALL="$ROOT/outputs/beat2/wide_256w.mp4"
ZOOM="$ROOT/outputs/beat2/wide_4w.mp4"

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
[ -f "$FONT" ] || FONT="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

mkdir -p "$(dirname "$OUT")"

# Speed up source clips so each segment lands ~target seconds.
# Wall src 240 frames @ 30fps = 8s. Want 5s wall display → 1.6x speed.
# Zoom src 240 frames @ 30fps = 8s. Want 8s zoom display → 1x speed.

ffmpeg -y \
  -f lavfi -i "color=c=0x0f172a:s=1920x1080:d=2.5:r=30" \
  -i "$WALL" \
  -i "$ZOOM" \
  -f lavfi -i "color=c=0x0f172a:s=1920x1080:d=2.5:r=30" \
  -filter_complex "\
[0:v]drawtext=fontfile=${FONT}:text='1024 robots, 1024 different targets':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=340,\
drawtext=fontfile=${FONT}:text='40cm x 40cm spread · 100\% success · 3.0 cm avg':fontcolor=0xa3b8e0:fontsize=36:x=(w-text_w)/2:y=460,\
drawtext=fontfile=${FONT}:text='one GPU · 40M PPO steps · EE-space actions':fontcolor=0x65a30d:fontsize=32:x=(w-text_w)/2:y=520,\
setsar=1,format=yuv420p[title];\
[1:v]setpts=PTS/1.6,fps=30,trim=duration=5,setpts=PTS-STARTPTS,\
scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0f172a,\
drawbox=x=0:y=0:w=iw:h=110:color=0x0f172a@0.85:t=fill,\
drawtext=fontfile=${FONT}:text='256 of 1024 · each target a different xy · all reach':fontcolor=white:fontsize=38:x=(w-text_w)/2:y=38,\
setsar=1,format=yuv420p[wall];\
[2:v]fps=30,trim=duration=8,setpts=PTS-STARTPTS,\
scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0f172a,\
drawbox=x=0:y=0:w=iw:h=110:color=0x0f172a@0.85:t=fill,\
drawtext=fontfile=${FONT}:text='4-world zoom · each arm goes to its own target':fontcolor=white:fontsize=38:x=(w-text_w)/2:y=38,\
setsar=1,format=yuv420p[zoom];\
[3:v]drawtext=fontfile=${FONT}:text='RoboSandbox':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=420,\
drawtext=fontfile=${FONT}:text='github.com/amarrmb/robosandbox':fontcolor=0x93c5fd:fontsize=38:x=(w-text_w)/2:y=540,\
setsar=1,format=yuv420p[outro];\
[title][wall][zoom][outro]concat=n=4:v=1:a=0[v]" \
  -map "[v]" \
  -c:v libx264 -profile:v main -pix_fmt yuv420p -preset medium -crf 20 \
  -movflags +faststart \
  "$OUT"

echo "wrote $OUT"
ls -lh "$OUT"
