import pandas as pd


class BaseDatasetAdapter:
    def preprocess(self, df):
        raise NotImplementedError

    def get_target_column(self):
        raise NotImplementedError


class TelcoAdapter(BaseDatasetAdapter):

    def get_target_column(self):
        return "Churn"

    def preprocess(self, df):

        # Robust drop 
        df = df.drop('customerID', axis=1, errors='ignore')

        # Encode target if exists
        if 'Churn' in df.columns and df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

        # Fix TotalCharges if exists
        if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == 'object':
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        return df




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
