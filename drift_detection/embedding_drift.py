import numpy as np
from scipy.stats import entropy


def softmax_drift(model, X_ref, X_cur):
    ref_probs = model.predict_proba(X_ref)
    cur_probs = model.predict_proba(X_cur)

    ref_mean = ref_probs.mean(axis=0)
    cur_mean = cur_probs.mean(axis=0)

    return entropy(ref_mean, cur_mean)


def prediction_space_drift(model, X_ref, X_cur):
    ref_pred = model.predict(X_ref)
    cur_pred = model.predict(X_cur)

    return float(np.abs(ref_pred.mean() - cur_pred.mean()))


def embedding_shift(model, X_ref, X_cur):
    results = {}

    # Classification models
    if hasattr(model, "predict_proba"):
        results["softmax_kl_drift"] = softmax_drift(model, X_ref, X_cur)

    # Works for both regression and classification
    results["prediction_mean_shift"] = prediction_space_drift(model, X_ref, X_cur)

    return results
