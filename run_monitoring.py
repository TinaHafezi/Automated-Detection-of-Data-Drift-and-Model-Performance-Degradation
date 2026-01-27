from dataset_adapter import TelcoAdapter  # ← بعداً می‌تونی EthereumAdapter بذاری
from data_loader import DataLoader
from drift_detection import calculate_psi

print("\n🔄 Loading data...")

adapter = TelcoAdapter()
loader = DataLoader(adapter)

ref_X, ref_y = loader.load_reference_data()
cur_X, cur_y = loader.load_current_data()


#### for changing the data and seeing a drift
# cur_X["MonthlyCharges"] *= 2

print("✅ Data loaded")

print("\n📊 Running drift detection...\n")

drifted_features = []

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

print("\n==============================")
print(f"TOTAL DRIFTED FEATURES: {len(drifted_features)} / {len(ref_X.columns)}")

if drifted_features:
    print("⚠️ DATA DRIFT DETECTED!")
else:
    print("✅ Data distribution stable.")
