import numpy as np


def feature_importance_shift(model, feature_names):

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()

    else:
        return {"explainability_shift": 0}

    # Normalize importance distribution
    importances = importances / (importances.sum() + 1e-10)

    # Drift proxy = variance of importance distribution
    shift_score = float(np.var(importances))

    return {"explainability_shift_score": shift_score}
