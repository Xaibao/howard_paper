#!/bin/bash
HLS_DIR="/home/t1204-3060/Howard/spectral_monitor/rtmp-server/media/live/stream"
mkdir -p "$HLS_DIR"

echo "啟動 ffmpeg HLS 轉換..."
/usr/bin/ffmpeg -i rtmp://localhost:1935/live/stream \
  -c:v libx264 -preset veryfast -tune zerolatency \
  -b:v 3000k -maxrate 3000k -bufsize 6000k \
  -c:a aac -b:a 64k \
  -f hls \
  -hls_time 1 \
  -hls_list_size 3 \
  -hls_flags delete_segments \
  "$HLS_DIR/index.m3u8" 2>&1
