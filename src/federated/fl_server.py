"""
FL Server — 跑在 3060 實驗室伺服器
FedAvg 聚合各縣市 Edge 端的 model weights
每輪結束後儲存 global model，記錄精確度

啟動：
  conda activate spectral
  python src/federated/fl_server.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import flwr as fl
from flwr.common import Metrics, Parameters, Scalar
from flwr.server.strategy import FedAvg

from federated.fl_model import get_model, TransformerFusionModel

# ── 路徑設定 ─────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent.parent
MODEL_PATH  = BASE_DIR / "models" / "mlp_transformer_v2.pth"
FL_MODEL_DIR = BASE_DIR / "models" / "fl_rounds"
FL_MODEL_DIR.mkdir(exist_ok=True)

LOG_PATH    = BASE_DIR / "outputs" / "fl_training_log.txt"
DEVICE      = "cpu"  # server 只做聚合，不需要 GPU
SERVER_ADDR = "0.0.0.0:8080"
NUM_ROUNDS  = 10

round_results: List[Dict] = []


def get_initial_parameters() -> Parameters:
    """載入預訓練模型作為 FL 初始 global model"""
    model = get_model(str(MODEL_PATH) if MODEL_PATH.exists() else None, device=DEVICE)
    weights = [val.cpu().numpy() for val in model.state_dict().values()]
    return fl.common.ndarrays_to_parameters(weights)


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """聚合各 client fit 回傳的 metrics"""
    total_samples = sum(n for n, _ in metrics)
    avg_loss = sum(n * m.get("train_loss", 0) for n, m in metrics) / total_samples
    return {"train_loss": avg_loss, "total_samples": total_samples}


def evaluate_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """聚合各 client evaluate 回傳的 accuracy"""
    total_samples = sum(n for n, _ in metrics)
    avg_acc = sum(n * m.get("accuracy", 0) for n, m in metrics) / total_samples
    return {"accuracy": avg_acc}


class SaveModelStrategy(FedAvg):
    """FedAvg + 每輪儲存 global model + 記錄 log"""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated

        parameters, metrics = aggregated

        # 把聚合後的 weights 存成 .pth
        weights = fl.common.parameters_to_ndarrays(parameters)
        model   = get_model(device=DEVICE)
        state   = model.state_dict()
        for k, w in zip(state.keys(), weights):
            state[k] = torch.tensor(w)
        model.load_state_dict(state)

        save_path = FL_MODEL_DIR / f"global_round_{server_round:02d}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"[Server] Round {server_round} model saved → {save_path.name}")

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated:
            loss, metrics = aggregated
            acc = metrics.get("accuracy", 0)
            log_line = f"Round {server_round:02d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}"
            print(f"[Server] {log_line}")
            with open(LOG_PATH, "a") as f:
                f.write(log_line + "\n")
            round_results.append({"round": server_round, "loss": loss, "accuracy": acc})
        return aggregated


def main():
    print(f"=== FOG 污染監測 FL Server ===")
    print(f"監聽地址：{SERVER_ADDR}")
    print(f"訓練輪數：{NUM_ROUNDS}")
    print(f"初始模型：{'預訓練模型' if MODEL_PATH.exists() else '隨機初始化'}")
    print(f"輸出路徑：{FL_MODEL_DIR}")

    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "w") as f:
        f.write("=== FL Training Log ===\n")

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        initial_parameters=get_initial_parameters(),
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
    )

    fl.server.start_server(
        server_address=SERVER_ADDR,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print(f"\n=== FL 訓練完成 ===")
    if round_results:
        best = max(round_results, key=lambda x: x["accuracy"])
        print(f"最佳輪次：Round {best['round']} | Accuracy: {best['accuracy']:.4f}")
        best_src  = FL_MODEL_DIR / f"global_round_{best['round']:02d}.pth"
        best_dest = BASE_DIR / "models" / "mlp_transformer_fl.pth"
        import shutil
        shutil.copy(best_src, best_dest)
        print(f"最佳 model 已存至：{best_dest}")


if __name__ == "__main__":
    main()
