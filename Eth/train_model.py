import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel

# Technical Indicators
def EMA(series, n):
    return series.ewm(span=n, min_periods=n).mean()

def ROC(series, n):
    return ((series.diff(n - 1) / series.shift(n - 1)) * 100)

def MOM(series, n):
    return series.diff(n)

def RSI(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def STOK(close, low, high, n):
    return ((close - low.rolling(n).min()) /
            (high.rolling(n).max() - low.rolling(n).min())) * 100

def STOD(close, low, high, n):
    return STOK(close, low, high, n).rolling(3).mean()


# TRAINING PIPELINE
def train_eth_model(data_path="Eth/eth_1h.csv"):
    print("📥 Loading ETH dataset...")
    df = pd.read_csv(data_path)

    # Rename columns to lowercase
    df.columns = df.columns.str.lower()

    # Use Date column
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # FEATURE ENGINEERING
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma60"] = df["close"].rolling(60).mean()

    df["ema10"] = EMA(df["close"], 10)
    df["ema30"] = EMA(df["close"], 30)
    df["ema200"] = EMA(df["close"], 200)

    df["roc10"] = ROC(df["close"], 10)
    df["roc30"] = ROC(df["close"], 30)

    df["mom10"] = MOM(df["close"], 10)
    df["mom30"] = MOM(df["close"], 30)

    df["rsi10"] = RSI(df["close"], 10)
    df["rsi30"] = RSI(df["close"], 30)
    df["rsi200"] = RSI(df["close"], 200)

    df["stok10"] = STOK(df["close"], df["low"], df["high"], 10)
    df["stod10"] = STOD(df["close"], df["low"], df["high"], 10)

    df["stok30"] = STOK(df["close"], df["low"], df["high"], 30)
    df["stod30"] = STOD(df["close"], df["low"], df["high"], 30)

    # Target = next close price
    df["target"] = df["close"].shift(-1)

    df = df.dropna()

    # Remove non-feature columns
    X = df.drop(["date", "symbol", "target"], axis=1)
    y = df["target"]

    # TIME SPLIT
    split = int(len(df) * 0.7)
    X_train, X_prod = X[:split], X[split:]
    y_train, y_prod = y[:split], y[split:]

    print("🌲 Feature selection...")
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)

    selector = SelectFromModel(base_model, prefit=True, threshold="median")
    selected_features = X.columns[selector.get_support()]

    X_train_sel = pd.DataFrame(selector.transform(X_train), columns=selected_features)
    X_prod_sel = pd.DataFrame(selector.transform(X_prod), columns=selected_features)

    print(f"Selected {len(selected_features)} features")

    print("🚀 Training final model...")
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_sel, y_train)

    preds = model.predict(X_prod_sel)

    baseline_metrics = {
        "mae": mean_absolute_error(y_prod, preds),
        "rmse": np.sqrt(mean_squared_error(y_prod, preds)),
        "r2": r2_score(y_prod, preds)
    }

    print("📊 Baseline metrics:", baseline_metrics)

    # SAVE ARTIFACTS
    pd.Series(X.columns).to_csv("Eth/all_features.csv", index=False)
    joblib.dump(model, "Eth/model.pkl")
    joblib.dump(selector, "Eth/feature_selector.pkl")
    pd.Series(selected_features).to_csv("Eth/selected_features.csv", index=False)
    pd.DataFrame([baseline_metrics]).to_csv("Eth/baseline_metrics.csv", index=False)

    pd.concat([X_train, y_train], axis=1).to_csv("Eth/reference.csv", index=False)
    pd.concat([X_prod, y_prod], axis=1).to_csv("Eth/current.csv", index=False)

    print("✅ ETH model training completed!")


if __name__ == "__main__":
    train_eth_model()
