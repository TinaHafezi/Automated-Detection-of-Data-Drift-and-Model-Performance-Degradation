from dataset_adapter import TelcoAdapter
from data_loader import DataLoader
from drift_detection import calculate_psi
import pandas as pd
from datetime import datetime
from pathlib import Path

print("\n🔄 Loading data...")

adapter = TelcoAdapter()
loader = DataLoader(adapter)

ref_X, ref_y = loader.load_reference_data()
cur_X, cur_y = loader.load_current_data()

print("✅ Data loaded")
print("\n📊 Running drift detection...\n")

drifted_features = []
results = []

for col in ref_X.columns:
    psi = calculate_psi(ref_X[col], cur_X[col])

    if psi < 0.1:
        status = "NO DRIFT"
    elif psi < 0.25:
        status = "MODERATE DRIFT"
        drifted_features.append(col)
    else:
        status = "SIGNIFICANT DRIFT"
        drifted_features.append(col)

    print(f"{col:25s} PSI={psi:.3f} → {status}")

    results.append({
        "timestamp": datetime.now(),
        "feature": col,
        "psi": round(psi, 4),
        "status": status
    })

print("\n" + "="*50)
print(f"TOTAL DRIFTED FEATURES: {len(drifted_features)} / {len(ref_X.columns)}")

if drifted_features:
    print("⚠️ DATA DRIFT DETECTED!")
else:
    print("✅ Data distribution stable.")

# SAVE TO EXCEL

log_file = Path("monitoring_log.xlsx")
df_results = pd.DataFrame(results)

if log_file.exists():
    old_df = pd.read_excel(log_file)
    df_results = pd.concat([old_df, df_results], ignore_index=True)

df_results.to_excel(log_file, index=False)

print(f"\n📁 Results appended to {log_file}")
