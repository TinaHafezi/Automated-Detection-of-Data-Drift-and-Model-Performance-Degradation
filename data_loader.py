import pandas as pd
import joblib
from pathlib import Path


class DataLoader:
    def __init__(self, adapter, base_path="Train model"):
        self.adapter = adapter
        self.base_path = Path(base_path)

        self.selector = joblib.load(self.base_path / "feature_selector.pkl")
        self.selected_features = pd.read_csv(
            self.base_path / "selected_features.csv"
        ).iloc[:, 0].tolist()

        self.all_features = pd.read_csv(
            self.base_path / "all_features.csv"
        ).iloc[:, 0].tolist()

    def _align_features(self, df):
        for col in self.all_features:
            if col not in df.columns:
                df[col] = 0
        df = df[self.all_features]
        return df

    def _apply_feature_selection(self, X):
        X_selected = self.selector.transform(X)
        return pd.DataFrame(X_selected, columns=self.selected_features)

    # 🔥 NOTE: reference.csv and current.csv are ALREADY PREPROCESSED
    def load_reference_data(self):
        df = pd.read_csv(self.base_path / "reference.csv")

        target_col = self.adapter.get_target_column()
        y = df[target_col]
        X = df.drop(target_col, axis=1)

        X = self._align_features(X)
        X_selected = self._apply_feature_selection(X)

        return X_selected, y

    def load_current_data(self):
        df = pd.read_csv(self.base_path / "current.csv")

        target_col = self.adapter.get_target_column()
        y = df[target_col]
        X = df.drop(target_col, axis=1)

        X = self._align_features(X)
        X_selected = self._apply_feature_selection(X)

        return X_selected, y
