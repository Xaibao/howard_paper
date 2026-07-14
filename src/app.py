"""
Spectral Monitor — Cloud Layer Flask Server
整合：Dashboard API + RAG + Claude API 污染分析
Port: 5000
"""
import os
import base64
import datetime
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── MLP-Transformer 模型定義 ─────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_size=1280, hidden_size=512, output_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerBranch(nn.Module):
    def __init__(self, output_size=128):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 512, kernel_size=40, stride=40)
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(512, output_size)
    def forward(self, x):
        x = self.patch_embed(x).flatten(2).permute(0, 2, 1)
        return self.fc(self.transformer(x).mean(dim=1))

class TransformerFusionModel(nn.Module):
    def __init__(self, mlp_input_size=1280, num_classes=5):
        super().__init__()
        self.mlp = MLP(input_size=mlp_input_size)
        self.transformer_branch = TransformerBranch()
        self.fc = nn.Linear(256, num_classes)
    def forward(self, txt, img):
        return self.fc(torch.cat((self.mlp(txt), self.transformer_branch(img)), dim=1))

CLASSES = ["motor_oil", "olive_oil", "palm_oil", "lard", "water"]
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((400, 1280)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

_infer_model = None
_infer_lock  = threading.Lock()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_infer_model():
    global _infer_model
    if _infer_model is None:
        with _infer_lock:
            if _infer_model is None:
                model_path = BASE_DIR.parent / "models" / "best_mlp_transformer_v2.pth"
                m = TransformerFusionModel().to(DEVICE)
                m.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
                m.eval()
                _infer_model = m
                print(f"[model] 載入完成：{model_path} on {DEVICE}")
    return _infer_model

app = Flask(__name__, static_folder="static")
CORS(app)

# ── 路徑設定 ────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
KB_DIR     = BASE_DIR / "knowledge_base"
DB_DIR     = BASE_DIR / "fog_expert_db"
IMAGE_DIR  = BASE_DIR / "static" / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── 分類對應（V2 五類）────────────────────────────────────────
CLASS_INFO = {
    "motor_oil": {
        "zh": "機油（廢棄引擎油）",
        "level": 3,
        "color": "#dc2626",
        "source": "工業/汽修廠",
    },
    "olive_oil": {
        "zh": "橄欖油",
        "level": 2,
        "color": "#d97706",
        "source": "餐飲業/食品廠",
    },
    "palm_oil": {
        "zh": "棕櫚油",
        "level": 2,
        "color": "#d97706",
        "source": "食品加工/棕櫚油廠",
    },
    "lard": {
        "zh": "豬油（動物脂肪）",
        "level": 2,
        "color": "#d97706",
        "source": "屠宰場/餐飲業",
    },
    "water": {
        "zh": "清水（無污染）",
        "level": 0,
        "color": "#16a34a",
        "source": "無",
    },
}

LEVEL_LABEL = {0: "正常", 1: "輕微污染", 2: "中度污染", 3: "嚴重污染（立即應變）"}

# ── RAG 向量資料庫（懶載入）────────────────────────────────────
_vector_db = None
_db_lock   = threading.Lock()

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        with _db_lock:
            if _vector_db is None:
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                _vector_db = Chroma(
                    persist_directory=str(DB_DIR),
                    embedding_function=embeddings,
                )
    return _vector_db

# ── 即時結果儲存（最多 50 筆）────────────────────────────────────
results_store = deque(maxlen=50)
store_lock    = threading.Lock()

# ── Ollama Llama 3.1 呼叫 ─────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

def call_llm(prompt: str) -> str:
    try:
        resp = http_requests.post(OLLAMA_URL, json={
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        }, timeout=120)
        return resp.json().get("response", "（無回應）")
    except Exception as e:
        return f"LLM 連線失敗：{e}"

# ── 污染分析 Prompt 工廠 ────────────────────────────────────────
def build_analysis_prompt(prediction: str, confidence: float, context: str) -> str:
    info = CLASS_INFO.get(prediction, {})
    zh_name  = info.get("zh", prediction)
    level    = info.get("level", 1)
    source   = info.get("source", "未知")
    level_label = LEVEL_LABEL.get(level, "未知")

    return f"""You are an expert in aquatic environmental pollution. Based on the spectral detection result and literature below, provide a comprehensive analysis in English with clear structure.

[Detection Result]
- Detected Substance: {zh_name} ({prediction})
- Confidence: {confidence:.1f}%
- Pollution Level: {level_label}
- Likely Source: {source}

[Reference Literature]
{context}

Please provide the following four sections, each separated by "## Title":

## 1. Pollution Analysis
Describe the characteristics of the substance, its hazard level, and ecological impact on water bodies.

## 2. Source Tracing
Identify the most likely pollution source and outline a source tracing investigation approach.

## 3. Problem Analysis
List specific risks this pollutant poses to the environment, public health, and infrastructure.

## 4. Treatment Recommendations
Provide immediate response measures and long-term remediation strategies, prioritized by pollution level.

Ensure the response is specific, professional, and actionable."""

# ═══════════════════════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════════════════════

# ── Dashboard：接收 Edge 端推送的辨識結果 ───────────────────────
@app.route("/api/report", methods=["POST"])
def report_result():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "Invalid data"}), 400

    image_url = None
    img_b64   = data.get("image_data")
    bmp_file  = data.get("bmp_file", "unknown.bmp")

    if img_b64:
        try:
            save_path = IMAGE_DIR / bmp_file
            save_path.write_bytes(base64.b64decode(img_b64))
            image_url = f"/static/images/{bmp_file}"
        except Exception as e:
            print(f"[image] 儲存失敗: {e}")

    prediction = data.get("prediction", "unknown")
    confidence = float(data.get("confidence", 0.0))
    info       = CLASS_INFO.get(prediction, {})

    entry = {
        "device":     data.get("device_id", "Unknown"),
        "prediction": prediction,
        "prediction_zh": info.get("zh", prediction),
        "level":      info.get("level", 1),
        "level_label": LEVEL_LABEL.get(info.get("level", 1), "未知"),
        "color":      info.get("color", "#6b7280"),
        "confidence": confidence,
        "txt_file":   data.get("txt_file", "N/A"),
        "bmp_file":   bmp_file,
        "image_url":  image_url,
        "timestamp":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with store_lock:
        results_store.appendleft(entry)

    return jsonify({"status": "success"})


# ── Dashboard：取得歷史結果 ─────────────────────────────────────
@app.route("/api/results", methods=["GET"])
def get_results():
    with store_lock:
        return jsonify(list(results_store))


# ── LLM：污染完整分析（污染+溯源+問題+處理）────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze_contamination():
    data       = request.json or {}
    prediction = data.get("prediction", "water")
    confidence = float(data.get("confidence", 0.0))

    # RAG 擷取相關文獻
    query = f"{prediction} water contamination treatment source analysis"
    try:
        db   = get_vector_db()
        docs = db.similarity_search(query, k=4)
        context = "\n\n".join(d.page_content for d in docs)
    except Exception as e:
        context = f"（知識庫查詢失敗：{e}）"

    prompt   = build_analysis_prompt(prediction, confidence, context)
    response = call_llm(prompt)

    return jsonify({
        "prediction":    prediction,
        "confidence":    confidence,
        "analysis":      response,
        "level":         CLASS_INFO.get(prediction, {}).get("level", 1),
        "level_label":   LEVEL_LABEL.get(CLASS_INFO.get(prediction, {}).get("level", 1), "未知"),
    })


# ── LLM：自由問答（RAG 輔助）──────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat_with_expert():
    data  = request.json or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"response": "請輸入問題"}), 400

    try:
        db   = get_vector_db()
        docs = db.similarity_search(query, k=3)
        context = "\n".join(d.page_content for d in docs)
    except Exception as e:
        context = f"（知識庫查詢失敗：{e}）"

    is_english = sum(1 for c in query if ord(c) < 128) / max(len(query), 1) > 0.85
    if is_english:
        system = "You are an expert in spectral analysis and water pollution control. Answer in English based on the provided literature, explaining spectral features and environmental impacts."
        prompt = f"{system}\n\n[Reference Literature]\n{context}\n\n[Question]\n{query}"
    else:
        system = "你是一位光譜分析與水體污染防治專家，請根據文獻資料以繁體中文回答，解釋光譜特徵與環境影響。"
        prompt = f"{system}\n\n【文獻資料】\n{context}\n\n【問題】\n{query}"
    return jsonify({"response": call_llm(prompt)})


# ── 靜態圖片 ─────────────────────────────────────────────────
@app.route("/static/images/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


# ── RPi Samba 即時資料 ────────────────────────────────────────
RPI_SHARE = Path("/home/t1204-3060/rpi_share")

@app.route("/api/rpi/latest", methods=["GET"])
def rpi_latest():
    bmp_files = sorted(RPI_SHARE.glob("*.bmp"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not bmp_files:
        return jsonify({"status": "no_data"}), 404
    latest_bmp = bmp_files[0]
    base = latest_bmp.stem
    sg_file  = RPI_SHARE / f"{base}_sg.txt"
    raw_file = RPI_SHARE / f"{base}_raw.txt"
    txt_name = sg_file.name if sg_file.exists() else (raw_file.name if raw_file.exists() else None)
    return jsonify({
        "status": "ok",
        "bmp_file": latest_bmp.name,
        "txt_file": txt_name,
        "timestamp": datetime.datetime.fromtimestamp(latest_bmp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    })

@app.route("/api/rpi/predict", methods=["GET"])
def rpi_predict():
    import time
    bmp_files = sorted(RPI_SHARE.glob("*.bmp"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not bmp_files:
        return jsonify({"status": "no_data"}), 404
    latest_bmp = bmp_files[0]
    base = latest_bmp.stem
    sg_file = RPI_SHARE / f"{base}_sg.txt"
    if not sg_file.exists():
        return jsonify({"status": "no_txt"}), 404
    try:
        t0 = time.time()
        txt = np.loadtxt(str(sg_file)).astype(np.float32) / 255.0
        img = IMG_TRANSFORM(Image.open(str(latest_bmp)).convert("RGB"))
        txt_t = torch.tensor(txt).unsqueeze(0).to(DEVICE)
        img_t = img.unsqueeze(0).to(DEVICE)
        model = get_infer_model()
        with torch.no_grad():
            logits = model(txt_t, img_t)
            probs  = torch.softmax(logits, dim=1)[0]
            pred_idx = probs.argmax().item()
            conf     = probs[pred_idx].item() * 100
        infer_ms = int((time.time() - t0) * 1000)
        prediction = CLASSES[pred_idx]
        info = CLASS_INFO.get(prediction, {})
        result = {
            "status":      "ok",
            "prediction":  prediction,
            "confidence":  round(conf, 1),
            "level":       info.get("level", 0),
            "level_label": LEVEL_LABEL.get(info.get("level", 0), "Normal"),
            "color":       info.get("color", "#6b7280"),
            "bmp_file":    latest_bmp.name,
            "txt_file":    sg_file.name,
            "timestamp":   datetime.datetime.fromtimestamp(latest_bmp.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "infer_ms":    infer_ms,
        }
        print(f"[predict] {prediction} {conf:.1f}% ({infer_ms}ms)")
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/rpi/bmp/<filename>")
def rpi_bmp(filename):
    return send_from_directory(str(RPI_SHARE), filename)

@app.route("/api/rpi/spectrum/<filename>")
def rpi_spectrum(filename):
    import numpy as np
    fpath = RPI_SHARE / filename
    if not fpath.exists():
        return jsonify({"error": "not found"}), 404
    try:
        data = np.loadtxt(str(fpath)).tolist()
        return jsonify({"data": data, "file": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 光譜 TXT 數據（回傳 1280 維陣列）────────────────────────
REAL_TEST_DIR = BASE_DIR.parent / "data_paper" / "real_source" / "real_test"

@app.route("/api/spectrum/<path:txt_file>")
def serve_spectrum(txt_file):
    import numpy as np
    # txt_file 格式：motor_oil/20260531_103514_sg.txt
    fpath = REAL_TEST_DIR / txt_file
    if not fpath.exists():
        return jsonify({"error": f"找不到 {txt_file}"}), 404
    try:
        data = np.loadtxt(str(fpath)).tolist()
        return jsonify({"data": data, "file": txt_file})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Demo：重新注入資料（展示用）──────────────────────────────
@app.route("/api/demo", methods=["POST"])
def inject_demo():
    with store_lock:
        results_store.clear()
    _init_demo_data()
    return jsonify({"status": "ok", "count": len(results_store)})


# ── 健康檢查 ──────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "timestamp": datetime.datetime.now().isoformat()})


def _init_demo_data():
    demo_data = [
        # 真實 motor_oil 量測資料（2026/05/31）
        ("motor_oil", 97.8, "Jetson-Edge-01", "20260531_103514"),
        ("motor_oil", 96.3, "Jetson-Edge-01", "20260531_103624"),
        ("motor_oil", 95.1, "Jetson-Edge-01", "20260531_103913"),
        ("motor_oil", 98.2, "Jetson-Edge-01", "20260531_103925"),
        ("motor_oil", 94.6, "Jetson-Edge-01", "20260531_103937"),
        # 真實 lard 量測資料（2026/05/31）
        ("lard",      91.3, "Jetson-Edge-01", "20260531_105004"),
        ("lard",      88.7, "Jetson-Edge-01", "20260531_105056"),
        ("lard",      93.2, "Jetson-Edge-01", "20260531_105108"),
        ("lard",      87.4, "Jetson-Edge-01", "20260531_105120"),
        ("lard",      90.1, "Jetson-Edge-01", "20260531_105132"),
        # 其他類別示範
        ("water",     99.1, "Jetson-Edge-01", "water_0001"),
        ("palm_oil",  72.4, "Jetson-Edge-01", "palm_oil_0001"),
        ("olive_oil", 91.2, "Jetson-Edge-01", "olive_oil_0001"),
    ]
    for i, (pred, conf, dev, fname) in enumerate(demo_data):
        info = CLASS_INFO.get(pred, {})
        ts = (datetime.datetime.now() - datetime.timedelta(seconds=i*30)).strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "device":        dev,
            "prediction":    pred,
            "prediction_zh": info.get("zh", pred),
            "level":         info.get("level", 1),
            "level_label":   LEVEL_LABEL.get(info.get("level", 1), "未知"),
            "color":         info.get("color", "#6b7280"),
            "confidence":    conf,
            "txt_file":      f"{fname}_sg.txt",
            "txt_path":      f"{pred}/{fname}_sg.txt",
            "bmp_file":      f"{fname}.bmp",
            "image_url":     f"/static/images/{fname}.bmp",
            "timestamp":     ts,
        }
        results_store.append(entry)
    print(f"[demo] 已注入 {len(demo_data)} 筆示範資料")

if __name__ == "__main__":
    _init_demo_data()
    print("Spectral Monitor Cloud Server 啟動於 http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
