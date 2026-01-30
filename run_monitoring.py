from data_loader import DataLoader
from drift_detection import detect_drift
from metrics_store import MetricsStore

print("🔄 Loading data...")
loader = DataLoader(dataset_name="telco")

ref_X, ref_y = loader.load_reference_data()
cur_X, cur_y = loader.load_current_data()

print("📊 Running drift detection...")
drift_results = detect_drift(ref_X, cur_X)

# Save to DB
store = MetricsStore()

metrics_to_save = {}
drift_count = 0

for feature, psi in drift_results.items():
    metrics_to_save[f"PSI_{feature}"] = psi
    if psi > 0.2:
        drift_count += 1

metrics_to_save["total_drifted_features"] = drift_count
metrics_to_save["total_features_checked"] = len(drift_results)

store.save_metrics(metrics_to_save)

print("\n" + "="*50)
print(f"TOTAL DRIFTED FEATURES: {drift_count} / {len(drift_results)}")
print("📁 Drift metrics saved to metrics.db")
