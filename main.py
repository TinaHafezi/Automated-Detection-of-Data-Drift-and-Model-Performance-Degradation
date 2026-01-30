from data_loader import DataLoader
from drift_detection import detect_drift
from model_monitor import ModelMonitor
from metrics_store import MetricsStore
from alert_system import AlertSystem
from config_loader import Config
import pandas as pd

print("=" * 60)
print("🚀 ML MONITORING PIPELINE STARTED")
print("=" * 60)

# LOAD CONFIG
config = Config()

DATASET_NAME = config.get("app", "dataset_name")
MODEL_PATH = config.get("model", "path")
BASELINE_PATH = config.get("model", "baseline_metrics_path")

DRIFT_THRESHOLD = config.get("monitoring", "drift_threshold")
PERF_DROP_THRESHOLD = config.get("monitoring", "performance_drop_threshold")

EMAIL_ENABLED = config.get("alerts", "email_enabled")
SENDER_EMAIL = config.get("alerts", "sender_email")
SENDER_PASS = config.get("alerts", "sender_password")
RECEIVER_EMAIL = config.get("alerts", "receiver_email")

# INIT SYSTEMS
loader = DataLoader(dataset_name=DATASET_NAME)
monitor = ModelMonitor()
store = MetricsStore()

# 1 LOAD DATA
print("\n🔄 Loading data...")
ref_X, ref_y = loader.load_reference_data()
cur_X, cur_y = loader.load_current_data()

# 2 DRIFT DETECTION
print("\n📊 Running drift detection...")
drift_results = detect_drift(ref_X, cur_X)

metrics_to_save = {}
drift_count = 0

for feature, psi in drift_results.items():
    metrics_to_save[f"PSI_{feature}"] = psi
    if psi > DRIFT_THRESHOLD:
        drift_count += 1

metrics_to_save["total_drifted_features"] = drift_count
metrics_to_save["total_features_checked"] = len(drift_results)

print(f"Drifted features: {drift_count}/{len(drift_results)}")

# 3 PERFORMANCE MONITORING
print("\n📈 Running performance monitoring...")

model = monitor.load_model(MODEL_PATH)
y_pred = monitor.predict(model, cur_X)

current_metrics = monitor.classification_metrics(cur_y, y_pred)
baseline_metrics = pd.read_csv(BASELINE_PATH).iloc[0].to_dict()
drops = monitor.compare_performance(baseline_metrics, current_metrics)

for k, v in current_metrics.items():
    metrics_to_save[f"current_{k}"] = v

for k, v in drops.items():
    metrics_to_save[f"drop_{k}"] = v

accuracy_drop = drops.get("accuracy", 0)

print("Current performance:", current_metrics)
print("Performance drops:", drops)

# 4 SAVE METRICS
store.save_metrics(metrics_to_save)
print("\n📁 Metrics saved to metrics.db")

# 5 ALERT LOGIC
print("\n🚨 Checking alert conditions...")

if EMAIL_ENABLED:
    alert = AlertSystem(SENDER_EMAIL, SENDER_PASS, RECEIVER_EMAIL)

    if drift_count > 0:
        alert.send_alert(
            "🚨 ML Alert: Data Drift Detected",
            f"{drift_count} features exceeded PSI threshold ({DRIFT_THRESHOLD})."
        )

    if accuracy_drop > PERF_DROP_THRESHOLD:
        alert.send_alert(
            "🚨 ML Alert: Model Performance Drop",
            f"Accuracy dropped by {accuracy_drop:.3f}"
        )

    if drift_count == 0 and accuracy_drop <= PERF_DROP_THRESHOLD:
        print("✅ System healthy. No alerts triggered.")

print("\n" + "=" * 50)
print("✅ PIPELINE FINISHED SUCCESSFULLY")
print("=" * 50)
