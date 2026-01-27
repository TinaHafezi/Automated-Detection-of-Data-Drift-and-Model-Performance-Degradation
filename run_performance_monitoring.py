from dataset_adapter import TelcoAdapter
from data_loader import DataLoader
from model_monitor import ModelMonitor
import pandas as pd
from datetime import datetime
from pathlib import Path

print("\n🔄 Loading model and data...")

adapter = TelcoAdapter()
loader = DataLoader(adapter)
monitor = ModelMonitor()

cur_X, cur_y = loader.load_current_data()

model = monitor.load_model("Train model/model.pkl")

baseline = pd.read_csv("Train model/baseline_metrics.csv").iloc[0].to_dict()

print("📈 Evaluating model...")

preds = monitor.predict(model, cur_X)
current_metrics = monitor.classification_metrics(cur_y, preds)

drops = monitor.compare_performance(baseline, current_metrics)

print("\nCurrent Metrics:", current_metrics)
print("Metric Drops:", drops)

status = "STABLE"
if any(v > 0.05 for v in drops.values()):
    status = "PERFORMANCE DEGRADATION ⚠️"

print("MODEL STATUS:", status)

# LOGGING 
log_file = Path("performance_log.xlsx")

row = pd.DataFrame([{
    "timestamp": datetime.now(),
    **baseline,
    **current_metrics,
    **drops,
    "status": status
}])

if log_file.exists():
    old = pd.read_excel(log_file)
    row = pd.concat([old, row], ignore_index=True)

row.to_excel(log_file, index=False)

print(f"\n📁 Logged to {log_file}")
