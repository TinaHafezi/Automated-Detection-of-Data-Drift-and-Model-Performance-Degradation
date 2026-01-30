import pandas as pd
import joblib
from dataset_adapter import get_adapter


class DataLoader:
    def __init__(self, dataset_name="telco"):
        self.dataset_name = dataset_name
        self.adapter = get_adapter(dataset_name)

        # Shared training artifacts
        self.selector = joblib.load("Train model/feature_selector.pkl")
        self.all_features = pd.read_csv("Train model/all_features.csv").iloc[:, 0].tolist()

    # ================= PUBLIC =================
    def load_reference_data(self):
        df = pd.read_csv("Train model/reference.csv")
        return self._process(df)

    def load_current_data(self):
        df = pd.read_csv("Train model/current.csv")
        return self._process(df)

    # ================= CORE PIPELINE =================
    def _process(self, df):
        target_col = self.adapter.get_target_column()

        y = df[target_col]
        X_raw = self.adapter.preprocess(df)

        # 🔹 Ensure SAME features as training
        X_raw = X_raw.reindex(columns=self.all_features, fill_value=0)

        # 🔹 Apply feature selection from training
        X_selected = self.selector.transform(X_raw)

        return pd.DataFrame(X_selected), y
