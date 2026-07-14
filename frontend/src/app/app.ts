import { Component, OnInit, OnDestroy, AfterViewChecked, AfterViewInit, ElementRef, ViewChild, NgZone, ChangeDetectorRef } from '@angular/core';
import Hls from 'hls.js';
import { CommonModule, DecimalPipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DomSanitizer, SafeHtml } from '@angular/platform-browser';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

const API = 'http://100.108.78.9:5000';

function mdToHtml(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^## (.+)$/gm, '<h4 style="color:#7c3aed;margin:10px 0 4px;font-size:12px;border-left:3px solid #7c3aed;padding-left:8px">$1</h4>')
    .replace(/^### (.+)$/gm, '<h5 style="color:#475569;margin:8px 0 3px;font-size:12px">$1</h5>')
    .replace(/^\* (.+)$/gm, '<li style="margin-left:16px;margin-bottom:3px">$1</li>')
    .replace(/(<li[\s\S]+?<\/li>)/g, '<ul style="margin:4px 0;padding:0">$1</ul>')
    .replace(/\n{2,}/g, '<br><br>')
    .replace(/\n/g, '<br>');
}

const CLASS_EN: Record<string, string> = {
  motor_oil: 'Motor Oil',
  olive_oil:  'Olive Oil',
  palm_oil:   'Palm Oil',
  lard:       'Lard',
  water:      'Water (Clean)',
};

const LEVEL_EN: Record<number, string> = {
  0: 'Normal',
  1: 'Caution',
  2: 'Moderate',
  3: 'Severe',
};

const CLASS_COLOR: Record<string, string> = {
  motor_oil: '#dc2626',
  olive_oil:  '#d97706',
  palm_oil:   '#d97706',
  lard:   '#d97706',
  water:      '#16a34a',
};

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, DecimalPipe],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class AppComponent implements OnInit, OnDestroy, AfterViewChecked, AfterViewInit {
  @ViewChild('chatContainer') private chatContainer!: ElementRef;
  @ViewChild('droneVideo') private droneVideo!: ElementRef<HTMLVideoElement>;

  streamConnected = false;
  fpsValue = 0;
  private hls: Hls | null = null;
  private hlsUrl = 'http://100.108.78.9:18000/live/stream/index.m3u8';
  private hlsRetryTimer: any;
  private fpsTimer: any;
  private fpsLastFrames = 0;
  // 顯示模式
  displayMode    = 'video';
  subDisplayMode = 'bmp';
  currentAgent   = 'drone';

  // 辨識結果
  predictionClass   = '---';
  predictionZh      = 'Awaiting detection...';
  predictionColor   = '#8b949e';
  confidence        = '0.0%';
  currentLevel      = 0;
  currentLevelLabel = 'Normal';
  llmReady          = true;

  // 光譜資料
  bmpUrl          = '';
  selectedBmpName = '---';
  selectedTxtName = '---';
  selectedTxtPath = '';
  chart: any;

  // 監測紀錄
  monitorHistory: any[] = [];
  private historyTimer: any;

  // RPi 即時資料（Drone + Boat 共用）
  rpiTimestamp     = '';
  private rpiTimer: any;
  private rpiLastTs = '';
  private userSelected = false;

  // LIME
  showLime     = false;
  limeImageUrl = '';

  // 污染分析
  isAnalyzing    = false;
  analysisResult = '';
  analysisHtml: SafeHtml = '';

  // LLM 對話
  chatInput   = '';
  chatHistory: { role: string; text: string }[] = [];
  isThinking  = false;

  constructor(private sanitizer: DomSanitizer, private ngZone: NgZone, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.chatHistory.push({
      role: 'Expert',
      text: 'Hello! I am the <strong>FOG Water Pollution Spectral Analysis Expert</strong>, equipped with a RAG knowledge base of 7 pollution documents.<br>You may ask about: pollutant hazards, source tracing methods, emergency response recommendations, etc.',
    });
    this.startHistoryPolling();
  }

  ngAfterViewInit() {
    // 頁面載入後自動開始嘗試串流
    setTimeout(() => this.retryStream(), 500);
  }

  ngAfterViewChecked() {
    try { this.chatContainer?.nativeElement.scrollTo(0, 99999); } catch {}
  }

  ngOnDestroy() {
    clearInterval(this.historyTimer);
    clearInterval(this.hlsRetryTimer);
    clearInterval(this.fpsTimer);
    clearInterval(this.rpiTimer);
    if (this.chart) this.chart.destroy();
    if (this.hls) this.hls.destroy();
  }

  private startFpsCounter(video: HTMLVideoElement) {
    clearInterval(this.fpsTimer);
    this.fpsLastFrames = 0;
    this.fpsTimer = setInterval(() => {
      const q = (video as any).getVideoPlaybackQuality?.();
      if (q) {
        const frames = q.totalVideoFrames - this.fpsLastFrames;
        this.fpsLastFrames = q.totalVideoFrames;
        this.fpsValue = frames;
      }
    }, 1000);
  }

  startStream() {
    const video = this.droneVideo?.nativeElement;
    if (!video) { console.warn('video element not found'); return; }
    if (this.hls) { this.hls.destroy(); this.hls = null; }

    console.log('Starting HLS stream:', this.hlsUrl);

    if (Hls.isSupported()) {
      this.hls = new Hls({
        lowLatencyMode: true,
        liveSyncDurationCount: 2,
        liveMaxLatencyDurationCount: 4,
        maxBufferLength: 3,
        liveDurationInfinity: true,
      });
      this.hls.loadSource(this.hlsUrl);
      this.hls.attachMedia(video);
      this.hls.on(Hls.Events.MANIFEST_PARSED, () => {
        console.log('HLS manifest parsed, playing...');
        video.play().then(() => {
          this.streamConnected = true;
          clearInterval(this.hlsRetryTimer);
          this.startFpsCounter(video);
        }).catch(e => console.error('play error:', e));
      });
      this.hls.on(Hls.Events.ERROR, (_, data) => {
        console.error('HLS error:', data.type, data.details);
        if (data.fatal) {
          this.streamConnected = false;
          this.hls?.destroy();
          this.hls = null;
        }
      });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = this.hlsUrl;
      video.play();
      this.streamConnected = true;
    }
  }

  retryStream() {
    clearInterval(this.hlsRetryTimer);
    setTimeout(() => {
      this.startStream();
      this.hlsRetryTimer = setInterval(() => {
        if (!this.streamConnected) this.startStream();
      }, 5000);
    }, 300);
  }

  // ── 監測紀錄輪詢（每 3 秒）─────────────────────────────────────
  startHistoryPolling() {
    this.fetchHistory();
    this.historyTimer = setInterval(() => this.fetchHistory(), 3000);
  }

  fetchHistory() {
    fetch(`${API}/api/results`)
      .then(r => r.json())
      .then((data: any[]) => {
        this.monitorHistory = data;
        if (data.length > 0 && !this.userSelected && this.currentAgent !== 'ship') {
          const keyword = this.currentAgent === 'drone' ? 'drone' : 'ground';
          const match = data.find(r => (r.device ?? '').toLowerCase().includes(keyword)) ?? data[0];
          if (!match) return;

          this.updatePrediction(match.prediction, match.confidence ?? 0);
          this.currentLevel      = match.level ?? 0;
          this.currentLevelLabel = this.levelLabel(match.level ?? 0);
          this.bmpUrl            = match.image_url ? `${API}${match.image_url}` : '';
          this.selectedBmpName   = match.bmp_file ?? '---';
          this.selectedTxtName   = match.txt_file ?? '---';
          this.selectedTxtPath   = match.txt_path ?? '';
          if (this.subDisplayMode === 'txt' && match.txt_path) {
            this.initChart(match.txt_path);
          }
        }
      })
      .catch(() => {});
  }

  updatePrediction(cls: string, conf: number) {
    this.predictionClass = cls;
    this.predictionZh    = CLASS_EN[cls]   ?? cls;
    this.predictionColor = CLASS_COLOR[cls] ?? '#8b949e';
    this.confidence      = `${conf.toFixed(1)}%`;
  }

  classLabel(cls: string): string { return CLASS_EN[cls] ?? cls; }
  levelLabel(level: number): string { return LEVEL_EN[level] ?? `Level ${level}`; }

  // ── 污染分析（/api/analyze）→ 推入對話框 ─────────────────────
  runAnalysis() {
    if (this.isAnalyzing || this.predictionClass === '---') return;
    this.isAnalyzing = true;

    const conf = parseFloat(this.confidence) || 0;
    this.chatHistory.push({
      role: 'User',
      text: `Please perform a full pollution analysis for the current detection result: <b>${this.predictionZh}</b>, confidence ${this.confidence}`,
    });
    this.isThinking = true;

    fetch(`${API}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prediction: this.predictionClass, confidence: conf }),
    })
      .then(r => r.json())
      .then((data: any) => {
        this.currentLevel      = data.level ?? 0;
        this.currentLevelLabel = data.level_label ?? '正常';
        this.chatHistory.push({ role: 'Expert', text: mdToHtml(data.analysis ?? '（無回應）') });
        this.isAnalyzing = false;
        this.isThinking  = false;
      })
      .catch((e: any) => {
        this.chatHistory.push({ role: 'Expert', text: `Analysis failed: ${e}` });
        this.isAnalyzing = false;
        this.isThinking  = false;
      });
  }

  // ── LLM 對話（/api/chat）─────────────────────────────────────
  sendToAI() {
    if (!this.chatInput.trim() || this.isThinking) return;
    const userMsg   = this.chatInput.trim();
    this.chatHistory.push({ role: 'User', text: userMsg });
    this.chatInput  = '';
    this.isThinking = true;

    fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: userMsg }),
    })
      .then(r => r.json())
      .then((data: any) => {
        this.chatHistory.push({ role: 'Expert', text: mdToHtml(data.response ?? '（無回應）') });
        this.isThinking = false;
      })
      .catch((e: any) => {
        this.chatHistory.push({ role: 'System', text: `❌ Backend connection failed. Please ensure python src/app.py is running: ${e}` });
        this.isThinking = false;
      });
  }

  // ── 模式切換 ──────────────────────────────────────────────────
  setSubMode(mode: string) {
    if (mode !== 'txt' && this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
    this.subDisplayMode = mode;
    if (mode === 'txt') {
      setTimeout(() => {
        if (this.currentAgent === 'ship') {
          if (this.selectedTxtName && this.selectedTxtName !== '---') this.initRpiChart(this.selectedTxtName);
        } else {
          this.initChart(this.selectedTxtPath);
        }
      }, 150);
    }
  }

  switchAgent(agent: string) {
    this.currentAgent   = agent;
    this.userSelected   = false;
    this.bmpUrl         = '';
    clearInterval(this.rpiTimer);

    if (agent === 'ship') {
      this.predictionZh      = 'Awaiting new image...';
      this.predictionClass   = '---';
      this.predictionColor   = '#8b949e';
      this.confidence        = '0.0%';
      this.currentLevel      = 0;
      this.currentLevelLabel = 'Normal';
      this.rpiTimestamp      = '';
      this.rpiLastTs         = '';
      this.fetchRpiLatest();
      this.rpiTimer = setInterval(() => this.pollRpiNew(), 3000);
    } else {
      // drone / ground：從 monitorHistory 找最新一筆
      this.rpiTimestamp = '';
      const keyword = agent === 'drone' ? 'drone' : 'ground';
      const match = this.monitorHistory.find(r =>
        (r.device ?? '').toLowerCase().includes(keyword)
      ) ?? this.monitorHistory[0];
      if (match) {
        this.updatePrediction(match.prediction, match.confidence ?? 0);
        this.currentLevel      = match.level ?? 0;
        this.currentLevelLabel = this.levelLabel(match.level ?? 0);
        this.bmpUrl            = match.image_url ? `${API}${match.image_url}` : '';
        this.selectedBmpName   = match.bmp_file ?? '---';
        this.selectedTxtName   = match.txt_file ?? '---';
        this.selectedTxtPath   = match.txt_path ?? '';
      }
    }
  }

  pollRpiNew() {
    fetch(`${API}/api/rpi/latest`)
      .then(r => r.json())
      .then((d: any) => {
        if (d.status !== 'ok') return;
        if (d.timestamp !== this.rpiLastTs) {
          this.ngZone.run(() => {
            this.rpiLastTs    = d.timestamp;
            this.rpiTimestamp = d.timestamp;
            this.predictionZh = 'Running inference...';
            this.cdr.detectChanges();
          });
          this.fetchRpiLatest();
        }
      })
      .catch(() => {});
  }

  fetchRpiLatest() {
    fetch(`${API}/api/rpi/predict`)
      .then(r => r.json())
      .then((d: any) => {
        this.ngZone.run(() => {
          if (d.status !== 'ok') return;
          this.bmpUrl          = `${API}/api/rpi/bmp/${d.bmp_file}`;
          this.selectedBmpName = d.bmp_file;
          this.selectedTxtName = d.txt_file ?? '---';
          this.selectedTxtPath = d.txt_file ?? '';
          this.rpiTimestamp = d.timestamp;
          this.updatePrediction(d.prediction, d.confidence ?? 0);
          this.currentLevel      = d.level ?? 0;
          this.currentLevelLabel = this.levelLabel(d.level ?? 0);
          this.rpiLastTs = d.timestamp;
          if (this.subDisplayMode === 'txt' && d.txt_file) this.initRpiChart(d.txt_file);
          this.cdr.detectChanges();
        });
      })
      .catch(() => {});
  }

  initRpiChart(filename: string) {
    const canvas = document.getElementById('spectrumChart') as HTMLCanvasElement;
    if (!canvas) return;
    const existing = (Chart as any).getChart(canvas);
    if (existing) { existing.destroy(); this.chart = null; }
    const wavelengths = Array.from({ length: 1280 }, (_, i) => Math.round(300 + i * (1100 - 300) / 1279));
    fetch(`${API}/api/rpi/spectrum/${filename}`)
      .then(r => r.json())
      .then(d => {
        if (this.chart) {
          this.chart.data.datasets[0].data = d.data;
          this.chart.update({ duration: 600, easing: 'easeInOutQuart' });
        } else {
          this.chart = new Chart(canvas, {
            type: 'line',
            data: {
              labels: wavelengths,
              datasets: [{ label: 'Intensity (SG)', data: d.data, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.08)', borderWidth: 1.5, pointRadius: 0, tension: 0.2 }],
            },
            options: {
              responsive: true, maintainAspectRatio: false,
              animation: { duration: 600, easing: 'easeInOutQuart' },
              scales: {
                x: { ticks: { color: '#64748b', maxTicksLimit: 10, font: { size: 10 } }, grid: { color: '#e2e8f0' } },
                y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#e2e8f0' } },
              },
              plugins: { legend: { display: false } },
            },
          });
        }
      })
      .catch(() => {});
  }

  onDataChange(event: Event) {
    const idx = parseInt((event.target as HTMLSelectElement).value, 10);
    const r   = this.monitorHistory[idx];
    if (!r) return;
    this.userSelected      = idx !== 0;
    this.bmpUrl            = r.image_url ? `${API}${r.image_url}` : '';
    this.selectedBmpName   = r.bmp_file ?? '---';
    this.selectedTxtName   = r.txt_file ?? '---';
    this.selectedTxtPath   = r.txt_path ?? '';
    this.updatePrediction(r.prediction, r.confidence ?? 0);
    this.currentLevel      = r.level ?? 0;
    this.currentLevelLabel = this.levelLabel(r.level ?? 0);
    if (this.subDisplayMode === 'txt') this.initChart(this.selectedTxtPath);
  }

  // ── Chart.js 光譜折線圖 ───────────────────────────────────────
  initChart(txtPath?: string) {
    const canvas = document.getElementById('spectrumChart') as HTMLCanvasElement;
    if (!canvas) return;
    if (this.chart) { this.chart.destroy(); this.chart = null; }
    const existing = (Chart as any).getChart(canvas);
    if (existing) existing.destroy();

    const wavelengths = Array.from({ length: 1280 }, (_, i) => Math.round(300 + i * (1100 - 300) / 1279));

    const drawChart = (data: number[]) => {
      if (this.chart) { this.chart.destroy(); this.chart = null; }
      this.chart = new Chart(canvas, {
        type: 'line',
        data: {
          labels: wavelengths,
          datasets: [{
            label: 'Intensity (SG filtered)',
            data,
            borderColor: '#7c3aed',
            backgroundColor: 'rgba(124,58,237,0.08)',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.2,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#64748b', maxTicksLimit: 10, font: { size: 10 } }, grid: { color: '#e2e8f0' } },
            y: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#e2e8f0' } },
          },
          plugins: { legend: { display: false } },
        },
      });
    };

    if (txtPath) {
      fetch(`${API}/api/spectrum/${txtPath}`)
        .then(r => r.json())
        .then(d => drawChart(d.data))
        .catch(() => drawChart(wavelengths.map(() => 0)));
    } else {
      drawChart(wavelengths.map(i => 0.5 + 0.3 * Math.sin(i / 50) + Math.random() * 0.05));
    }
  }
}
