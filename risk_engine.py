import numpy as np

class RiskEngine:

    def __init__(self):
        self.weights = {
            "statistical_drift": 0.35,
            "data_quality": 0.15,
            "embedding_drift": 0.20,
            "performance_drop": 0.30
        }

    def normalize(self, value, max_expected):
        return min(value / max_expected, 1.0)

    def compute_risk(
        self,
        num_drifted_features,
        total_features,
        data_quality_issues,
        embedding_score,
        performance_drop
    ):
        # 1️⃣ Statistical Drift Risk
        drift_ratio = num_drifted_features / max(total_features, 1)
        drift_risk = drift_ratio

        # 2️⃣ Data Quality Risk
        dq_risk = self.normalize(data_quality_issues, 10)

        # 3️⃣ Embedding / Prediction Space Drift
        emb_risk = self.normalize(embedding_score, 1)

        # 4️⃣ Performance Drop Risk
        perf_risk = self.normalize(performance_drop, 1)

        final_score = (
            drift_risk * self.weights["statistical_drift"] +
            dq_risk * self.weights["data_quality"] +
            emb_risk * self.weights["embedding_drift"] +
            perf_risk * self.weights["performance_drop"]
        )

        return round(final_score * 100, 2)
