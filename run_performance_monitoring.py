from model_monitor import ModelMonitor
from data_loader import DataLoader
from metrics_store import MetricsStore
import joblib

print("🔄 Loading model & data...")

loader = DataLoader(dataset_name="telco")
X_current, y_current = loader.load_current_data()

monitor = ModelMonitor()
model = monitor.load_model("model.pkl")

y_pred = monitor.predict(model, X_current)

print("📈 Calculating performance metrics...")

# فرض: مدل طبقه‌بندی Telco
current_metrics = monitor.classification_metrics(y_current, y_pred)

# Baseline metrics (ذخیره‌شده از زمان train)
baseline_metrics = {
    "accuracy": 0.80,
    "f1": 0.78,
    "precision": 0.77,
    "recall": 0.79,
}

drops = monitor.compare_performance(baseline_metrics, current_metrics)

# Save to DB
store = MetricsStore()

metrics_to_save = {}

for k, v in current_metrics.items():
    metrics_to_save[f"current_{k}"] = v

for k, v in drops.items():
    metrics_to_save[f"drop_{k}"] = v

store.save_metrics(metrics_to_save)

print("\n" + "="*50)
print("Current Performance:", current_metrics)
print("Performance Drops:", drops)
print("📁 Performance metrics saved to metrics.db")
