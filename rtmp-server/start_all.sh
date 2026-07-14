#!/bin/bash
cd /home/t1204-3060/Howard/spectral_monitor/rtmp-server

# 清掉所有相關 process
kill -9 $(lsof -ti:1935) 2>/dev/null
kill -9 $(lsof -ti:18000) 2>/dev/null
kill -9 $(pgrep -f "start_hls") 2>/dev/null
kill -9 $(pgrep -f "hls_http") 2>/dev/null
sleep 2

mkdir -p media/live/stream

# 1. RTMP server (port 1935)
node server.js > /tmp/rtmp.log 2>&1 &
echo "[1] RTMP server PID: $!"
sleep 3

# 2. ffmpeg RTMP→HLS 轉換（stream copy，不重新編碼，速度快）
/usr/bin/ffmpeg -i rtmp://localhost:1935/live/stream \
  -c:v copy \
  -an \
  -f hls -hls_time 1 -hls_list_size 5 \
  -hls_flags delete_segments+split_by_time \
  media/live/stream/index.m3u8 > /tmp/hls.log 2>&1 &
echo "[2] ffmpeg HLS PID: $!"

# 3. HTTP server serve HLS (port 12000)
python3 - > /tmp/hls_http.log 2>&1 << 'PYEOF' &
import http.server, socketserver, os
os.chdir('/home/t1204-3060/Howard/spectral_monitor/rtmp-server/media')
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Cache-Control','no-cache')
        super().end_headers()
    def log_message(self, *a): pass
with socketserver.TCPServer(('0.0.0.0', 18000), H) as s:
    print('HLS HTTP server on :18000')
    s.serve_forever()
PYEOF
echo "[3] HLS HTTP PID: $!"

echo ""
echo "=== 全部啟動完成 ==="
echo "RTMP 推流：rtmp://100.108.78.9:1935/live/stream"
echo "HLS  播放：http://100.108.78.9:18000/live/stream/index.m3u8"
