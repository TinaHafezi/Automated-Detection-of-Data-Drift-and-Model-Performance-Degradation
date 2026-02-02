# train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score

def train_pipeline(path="Train model/telco.csv"):
    df = pd.read_csv(path)

    df = df.drop("customerID", axis=1, errors="ignore")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    if df["TotalCharges"].dtype == "object":
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True)
    y = df["Churn"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_prod, y_val, y_prod = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    base = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", max_depth=10)
    base.fit(X_train, y_train)

    selector = SelectFromModel(base, prefit=True, threshold="median")

    selected_features = X.columns[selector.get_support()]

    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)
    X_prod_sel = selector.transform(X_prod)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", max_depth=10)
    model.fit(X_train_sel, y_train)

    val_pred = model.predict(X_val_sel)

    baseline = {
        "accuracy": accuracy_score(y_val, val_pred),
        "f1": f1_score(y_val, val_pred)
    }

    # SAVE CLEAN ARTIFACTS
    joblib.dump(model, "Train model/model.pkl")
    joblib.dump(selector, "Train model/feature_selector.pkl")

    pd.Series(X.columns).to_csv("Train model/all_features.csv", index=False, header=False)
    pd.Series(selected_features).to_csv("Train model/selected_features.csv", index=False, header=False)

    pd.concat([X_train, y_train], axis=1).to_csv("Train model/reference.csv", index=False)
    pd.concat([X_prod, y_prod], axis=1).to_csv("Train model/current.csv", index=False)

    pd.DataFrame([baseline]).to_csv("Train model/baseline_metrics.csv", index=False)

    print("✅ Training complete — artifacts rebuilt CLEAN")

if __name__ == "__main__":
    train_pipeline()
