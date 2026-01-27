import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelMonitor:

    def load_model(self, path):
        return joblib.load(path)

    def predict(self, model, X):
        return model.predict(X)

    def classification_metrics(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

    def regression_metrics(self, y_true, y_pred):
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

    def compare_performance(self, train_metrics, current_metrics):
        drops = {}
        for key in train_metrics:
            drops[key] = train_metrics[key] - current_metrics.get(key, 0)
        return drops
