# 4060_SETUP.md — 4060 電腦設定指南（Windows）

> Tailscale IP：`100.125.219.94`
> 角色：Cloud Angular 前端 UI
> Flask API 跑在 3060（`100.108.78.9:5000`），4060 只需要跑前端

---

## 1. 確認 3060 Flask API 可用

在 4060 或任何有 Tailscale 的機器執行：

```powershell
curl http://100.108.78.9:5000/api/health
curl http://100.108.78.9:5000/api/results
```

---

## 2. Angular 前端 UI

**環境要求**：Node.js 18+、Angular CLI 17+

**安裝：**
```powershell
npm install -g @angular/cli
cd C:\path\to\spectral_monitor\frontend
npm install
```

**前端連到 3060 Flask API：**
```typescript
// environment.ts
export const environment = {
  apiUrl: 'http://100.108.78.9:5000/api'
};
```

**開發模式啟動（port 4200）：**
```powershell
ng serve --host 0.0.0.0 --port 4200
```

**生產模式：**
```powershell
ng build --configuration production
```

---

## 3. 4060 Claude Code 的任務

Angular 前端需要完成：

1. **監測資料頁面**
   - 呼叫 `GET /api/results` 顯示最新 50 筆
   - 污染等級顏色標示（Level 0 綠 / Level 2 橘 / Level 3 紅）
   - 即時更新（每 5 秒 polling）

2. **污染分析頁面**
   - 呼叫 `POST /api/analyze`，傳入辨識結果
   - 顯示 LLM 回傳的四段分析（污染分析 / 溯源 / 問題 / 處理建議）

3. **LLM 問答頁面**
   - 呼叫 `POST /api/chat`，傳入 `{"message": "..."}`
   - 顯示對話介面

**API 格式參考：**
```json
// POST /api/chat
{ "message": "水中發現機油污染應如何處理？" }

// POST /api/analyze
{ "prediction": "motor_oil", "confidence": 95.2 }

// GET /api/results 回傳
[{ "timestamp": "...", "prediction": "motor_oil", "confidence": 95.2, "level": 3 }, ...]
```

---

## 4. Tailscale 連線確認

```powershell
# 4060 ping 3060
ping 100.108.78.9
```

```bash
# 3060 ping 4060
ping 100.125.219.94
```

---

## 5. 注意事項

- LLM 已改為 Claude API，**不需要在 4060 跑 Ollama**
- Flask CORS 已開啟，Angular 可直接跨域呼叫 3060
- 3060 Flask API 必須先啟動，Angular 才能正常顯示資料
