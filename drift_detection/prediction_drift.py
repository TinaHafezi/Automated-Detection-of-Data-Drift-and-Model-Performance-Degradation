import numpy as np
from scipy.stats import wasserstein_distance

def prediction_mean_shift(model, X_ref, X_cur):
    ref_pred = model.predict(X_ref)
    cur_pred = model.predict(X_cur)
    return abs(ref_pred.mean() - cur_pred.mean())

def prediction_distribution_shift(model, X_ref, X_cur):
    ref_pred = model.predict(X_ref)
    cur_pred = model.predict(X_cur)
    return wasserstein_distance(ref_pred, cur_pred)

def prediction_confidence_shift(model, X_ref, X_cur):
    if not hasattr(model, "predict_proba"):
        return 0  # regression model
    ref_conf = model.predict_proba(X_ref).max(axis=1)
    cur_conf = model.predict_proba(X_cur).max(axis=1)
    return abs(ref_conf.mean() - cur_conf.mean())
