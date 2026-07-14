const NodeMediaServer = require('node-media-server');

const config = {
  rtmp: {
    port: 1935,
    chunk_size: 600,
    gop_cache: false,
    ping: 30,
    ping_timeout: 60,
    publish_timeout: 1000,
    idle_timeout: 300
  },
  http: {
    host: '0.0.0.0',
    port: 18080,
    mediaroot: './media',
    allow_origin: '*'
  },
  trans: {
    ffmpeg: '/usr/bin/ffmpeg',
    tasks: [
      {
        app: 'live',
        hls: true,
        hlsFlags: '[hls_time=1:hls_list_size=3:hls_flags=delete_segments]',
        dash: false,
        ffmpegOptions: [
          '-c:v', 'libx264',
          '-preset', 'veryfast',
          '-tune', 'zerolatency',
          '-b:v', '3000k',
          '-maxrate', '3000k',
          '-bufsize', '6000k',
          '-an'
        ]
      }
    ]
  }
};

const nms = new NodeMediaServer(config);
nms.run();
console.log('RTMP: rtmp://100.108.78.9:1935/live/drone');
console.log('HLS:  http://100.108.78.9:12000/live/drone/index.m3u8');