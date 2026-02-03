import pandas as pd
import joblib

class DataLoader:
    def __init__(self, dataset_name="telco"):
        self.dataset_name = dataset_name

        if dataset_name == "telco":
            base_path = "Train model"
        elif dataset_name == "ethereum":
            base_path = "Eth"
        else:
            raise ValueError("Unknown dataset")

        self.base_path = base_path

        # Load artifacts from correct folder
        self.selector = joblib.load(f"{base_path}/feature_selector.pkl")
        self.all_features = pd.read_csv(f"{base_path}/all_features.csv").iloc[:, 0].tolist()
        self.selected_features = pd.read_csv(f"{base_path}/selected_features.csv").iloc[:, 0].tolist()

    def load_reference_data(self):
        df = pd.read_csv(f"{self.base_path}/reference.csv")
        return self._process(df)

    def load_current_data(self):
        df = pd.read_csv(f"{self.base_path}/current.csv")
        return self._process(df)

    def _process(self, df):
        target_col = "Churn" if self.dataset_name == "telco" else "target"

        y = df[target_col]
        X = df.drop(target_col, axis=1)

        X = X.reindex(columns=self.all_features, fill_value=0)

        X_selected_array = self.selector.transform(X)

        X_selected = pd.DataFrame(
            X_selected_array,
            columns=self.selected_features,
            index=X.index
        )

        return X_selected, y