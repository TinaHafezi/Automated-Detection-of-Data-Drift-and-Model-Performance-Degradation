import json
import math
import pandas as pd
from config_loader import Config
from risk_engine import RiskEngine
from data_loader import DataLoader
from model_monitor import ModelMonitor
from metrics_store import MetricsStore
from alert_system import AlertSystem
from drift_detection import run_full_drift_analysis

print("=" * 60)
print("🚀 ML MONITORING PIPELINE STARTED")
print("=" * 60)

# ================= CONFIG =================
config = Config()

DATASET_NAME = config.get("app", "dataset_name")
MODEL_PATH = config.get("model", "path")
BASELINE_PATH = config.get("model", "baseline_metrics_path")

DRIFT_THRESHOLD = float(config.get("monitoring", "drift_threshold"))
PERF_DROP_THRESHOLD = float(config.get("monitoring", "performance_drop_threshold"))

EMAIL_ENABLED = config.get("alerts", "email_enabled")
SENDER_EMAIL = config.get("alerts", "sender_email")
SENDER_PASS = config.get("alerts", "sender_password")
RECEIVER_EMAIL = config.get("alerts", "receiver_email")

# ================= INIT =================
loader = DataLoader(dataset_name=DATASET_NAME)
monitor = ModelMonitor()
store = MetricsStore()

model = monitor.load_model(MODEL_PATH)

# ================= LOAD DATA =================
print("\n🔄 Loading data...")
ref_X, ref_y = loader.load_reference_data()
cur_X, cur_y = loader.load_current_data()

# ================= DRIFT DETECTION =================
print("\n📊 Running drift detection...")

drift_results = run_full_drift_analysis(model, ref_X, cur_X)

metrics_to_save = {}
drift_count_static = 0
drift_count_dynamic = 0


def _get_dynamic_threshold(metric_name, base_threshold):
    """Compute a dynamic threshold based on historical mean + 3σ."""
    mean, std = store.get_historical_stats(metric_name)
    if mean is None or std is None or math.isnan(mean):
        return base_threshold
    return mean + 3 * std


for metric_name, value in drift_results.items():
    # Store dicts/lists as JSON strings to preserve information
    if isinstance(value, (dict, list)):
        metrics_to_save[metric_name] = json.dumps(value, ensure_ascii=False)
        continue

    # Convert numeric-like values to float
    try:
        num_val = float(value)
        metrics_to_save[metric_name] = num_val
    except Exception:
        metrics_to_save[metric_name] = str(value)
        continue

    # -------- STATIC DRIFT DETECTION --------
    if metric_name.startswith("KS_") or metric_name.startswith("WASS_") or metric_name.startswith("PSI_"):
        if metric_name.startswith("KS_"):
            # For KS, low p-value (< threshold) means drift
            if num_val < DRIFT_THRESHOLD:
                drift_count_static += 1
        else:
            # For Wasserstein/PSI, high value (> threshold) means drift
            if num_val > DRIFT_THRESHOLD:
                drift_count_static += 1

        # -------- DYNAMIC DRIFT DETECTION --------
        dyn_threshold = _get_dynamic_threshold(metric_name, DRIFT_THRESHOLD)
        if metric_name.startswith("KS_"):
            if num_val < dyn_threshold:
                drift_count_dynamic += 1
        else:
            if num_val > dyn_threshold:
                drift_count_dynamic += 1

metrics_to_save["static_drifted_features"] = drift_count_static
metrics_to_save["dynamic_drifted_features"] = drift_count_dynamic
metrics_to_save["total_features_checked"] = int(
    drift_results.get("TOTAL_FEATURES_CHECKED", len(ref_X.columns))
)

print(f"Static Drifted Features: {drift_count_static}")
print(f"Dynamic Drifted Features: {drift_count_dynamic}")

# ================= PERFORMANCE MONITORING =================
print("\n📈 Running performance monitoring...")

y_pred = monitor.predict(model, cur_X)

if DATASET_NAME.lower() == "ethereum":
    current_metrics = monitor.regression_metrics(cur_y, y_pred)
else:
    current_metrics = monitor.classification_metrics(cur_y, y_pred)

baseline_metrics = pd.read_csv(BASELINE_PATH).iloc[0].to_dict()
drops = monitor.compare_performance(baseline_metrics, current_metrics)

for k, v in current_metrics.items():
    try:
        metrics_to_save[f"current_{k}"] = float(v)
    except Exception:
        metrics_to_save[f"current_{k}"] = str(v)

for k, v in drops.items():
    try:
        metrics_to_save[f"drop_{k}"] = float(v)
    except Exception:
        metrics_to_save[f"drop_{k}"] = str(v)

print("Current performance:", current_metrics)
print("Performance drops:", drops)

# ================= DYNAMIC PERFORMANCE THRESHOLD =================
mean_perf, std_perf = store.get_historical_stats("current_rmse")

if mean_perf is not None and std_perf is not None:
    dynamic_perf_threshold = mean_perf + 2 * std_perf
else:
    dynamic_perf_threshold = PERF_DROP_THRESHOLD

metrics_to_save["dynamic_perf_threshold"] = dynamic_perf_threshold

# ================= RISK ENGINE =================
print("\n🧠 Calculating system risk score...")

risk_engine = RiskEngine()

embedding_score = drift_results.get("softmax_drift")
if embedding_score is None:
    embedding_score = drift_results.get("prediction_space_drift")
if embedding_score is None:
    embedding_score = drift_results.get("EMB_SHIFT", 0)
data_quality_issues = drift_results.get("DQ_TOTAL_NEW_CATEGORY_FEATURES", 0)

actual_perf_drop = max(abs(v) for v in drops.values() if isinstance(v, (int, float)))
risk_score = risk_engine.compute_risk(
    num_drifted_features=drift_count_dynamic,
    total_features=len(ref_X.columns),
    data_quality_issues=data_quality_issues,
    embedding_score=embedding_score,
    performance_drop=actual_perf_drop,
)

metrics_to_save["system_risk_score"] = risk_score

print(f"🚨 SYSTEM RISK SCORE: {risk_score:.2f}/100")

# ================= SAVE METRICS =================
store.save_metrics(metrics_to_save)
print("\n📁 Metrics saved to metrics.db")

# ================= ALERTS =================
print("\n🚨 Checking alert conditions...")

if EMAIL_ENABLED:
    alert = AlertSystem(SENDER_EMAIL, SENDER_PASS, RECEIVER_EMAIL)

    if risk_score > 70:
        alert.send_alert(
            "🔥 CRITICAL ML RISK",
            f"System Risk Score reached {risk_score:.1f}/100",
        )
    elif risk_score > 40:
        alert.send_alert(
            "⚠️ ML System Warning",
            f"Elevated risk detected: {risk_score:.1f}/100",
        )

print("\n" + "=" * 60)
print("✅ PIPELINE FINISHED SUCCESSFULLY")
print("=" * 60)
