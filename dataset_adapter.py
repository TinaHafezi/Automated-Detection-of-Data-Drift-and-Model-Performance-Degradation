import pandas as pd

class BaseDatasetAdapter:
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_target_column(self) -> str:
        raise NotImplementedError


# ================= TELCO =================
class TelcoAdapter(BaseDatasetAdapter):

    def preprocess(self, df):
        df = df.drop('customerID', axis=1)

        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        if df['TotalCharges'].dtype == 'object':
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        X = df.drop('Churn', axis=1)
        X = pd.get_dummies(X, drop_first=True)
        return X

    def get_target_column(self):
        return "Churn"


# ================= ETHEREUM =================
class EthereumAdapter(BaseDatasetAdapter):

    def preprocess(self, df):
        df['price_change'] = df['close'].pct_change()
        df = df.dropna()

        X = df.drop('price_change', axis=1)
        return X

    def get_target_column(self):
        return "price_change"


# Adapter factory
def get_adapter(dataset_name: str) -> BaseDatasetAdapter:
    if dataset_name == "telco":
        return TelcoAdapter()
    elif dataset_name == "ethereum":
        return EthereumAdapter()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
