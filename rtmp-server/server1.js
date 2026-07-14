const NodeMediaServer = require('node-media-server');

const config = {
  rtmp: {
    port: 1935,
    chunk_size: 600,         // 減小 chunk_size，降低延遲
    gop_cache: false,
    ping: 30,
    ping_timeout: 60,
    publish_timeout: 1000,
    idle_timeout: 300
  },
  http: {
    host: '0.0.0.0',         // 新增 host 設定，避免顯示 undefined
    port: 12000,
    mediaroot: './media',
    allow_origin: '*'        // 確保允許跨域訪問
  },
  trans: {
    ffmpeg: '/usr/bin/ffmpeg', // Termux 中的 ffmpeg 路徑
    tasks: [
      {
        app: 'live',       // 單一應用名稱
        hls: true,
        hlsFlags: '[hls_time=1:hls_list_size=3:hls_flags=delete_segments]', // 調整 hls_time 以降低延遲
        dash: false,
        ffmpegOptions: [
          '-c:v', 'libx264',       // 使用 H.264 編碼器以提升兼容性和畫質
          '-preset', 'veryfast',    // 快速編碼以減少延遲
          '-tune', 'zerolatency',   // 調整編碼器以降低延遲
          '-b:v', '3000k',          // 視訊比特率，可根據需求調整
          '-maxrate', '3000k',
          '-bufsize', '6000k',
          '-an'                     // 移除音訊
        ]
      }
    ]
  }
};

const nms = new NodeMediaServer(config);
nms.run();
