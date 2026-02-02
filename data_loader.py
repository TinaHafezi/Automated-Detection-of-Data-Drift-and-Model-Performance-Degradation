import pandas as pd
import joblib

class DataLoader:
    def __init__(self, dataset_name="telco"):
        self.selector = joblib.load("Train model/feature_selector.pkl")

        self.all_features = pd.read_csv(
            "Train model/all_features.csv", header=None
        ).iloc[:, 0].tolist()

        self.selected_features = pd.read_csv(
            "Train model/selected_features.csv", header=None
        ).iloc[:, 0].tolist()

    def load_reference_data(self):
        df = pd.read_csv("Train model/reference.csv")
        return self._process(df)

    def load_current_data(self):
        df = pd.read_csv("Train model/current.csv")
        return self._process(df)

    def _process(self, df):
        y = df["Churn"]
        X = df.drop("Churn", axis=1)

        # Align columns exactly as training
        X = X.reindex(columns=self.all_features, fill_value=0)

        # Feature selection
        X_selected_array = self.selector.transform(X)

        # Reattach real feature names
        X_selected = pd.DataFrame(
            X_selected_array,
            columns=self.selected_features,
            index=X.index
        )

        return X_selected, y
