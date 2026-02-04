from .statistical_drift import ks_drift, wasserstein_drift
from .data_quality import (
    check_missing_drift,
    check_zero_drift,
    detect_new_categories
)
from .embedding_drift import embedding_shift
from .explainability_shift import feature_importance_shift
from .prediction_drift import (
    prediction_mean_shift,
    prediction_distribution_shift,
    prediction_confidence_shift
)
import json
from .statistical_drift import ks_drift, wasserstein_drift
from .data_quality import (
    check_missing_drift,
    check_zero_drift,
    detect_new_categories
)
from .embedding_drift import embedding_shift
from .explainability_shift import feature_importance_shift
from .prediction_drift import (
    prediction_mean_shift,
    prediction_distribution_shift,
    prediction_confidence_shift
)

def _safe_number(x):
    """Try to coerce to float; otherwise return original."""
    try:
        return float(x)
    except Exception:
        return x

def run_full_drift_analysis(model, ref_X, cur_X):
    results = {}

    # ---------- Statistical Drift (per-feature) ----------
    for col in ref_X.columns:
        # produce standardized metric names -> uppercase prefixes
        ks_name = f"KS_{col}"
        wass_name = f"WASS_{col}"

        ks_val = ks_drift(ref_X[col], cur_X[col])
        wass_val = wasserstein_drift(ref_X[col], cur_X[col])

        results[ks_name] = _safe_number(ks_val)
        results[wass_name] = _safe_number(wass_val)

    # ---------- Data Quality ----------
    missing_drift = check_missing_drift(ref_X, cur_X)    # expected dict: {feature: fraction_missing_diff}
    zero_drift = check_zero_drift(ref_X, cur_X)          # expected dict: {feature: fraction_zero_diff}
    new_categories = detect_new_categories(ref_X, cur_X) # expected dict: {feature: [new_cats]}

    # Flatten DQ outputs with standardized names
    for feature, val in missing_drift.items():
        results[f"DQ_MISSING_{feature}"] = _safe_number(val)

    for feature, val in zero_drift.items():
        results[f"DQ_ZERO_{feature}"] = _safe_number(val)

    # Store number of new categories and list as JSON (do not lose info)
    for feature, newcats in new_categories.items():
        results[f"DQ_NEWCATS_COUNT_{feature}"] = _safe_number(len(newcats))
        results[f"DQ_NEWCATS_LIST_{feature}"] = json.dumps(list(newcats), ensure_ascii=False)

    # Add aggregate DQ counts for convenience
    results["DQ_TOTAL_NEW_CATEGORY_FEATURES"] = _safe_number(sum(1 for v in new_categories.values() if v))
    results["DQ_TOTAL_FEATURES"] = _safe_number(len(ref_X.columns))

    # ---------- Representation / Embedding Drift ----------
    # embedding_shift may return a scalar or dict -> flatten
    emb = embedding_shift(model, ref_X, cur_X)
    if isinstance(emb, dict):
        for k, v in emb.items():
            results[f"EMB_{k}"] = _safe_number(v)
    else:
        results["EMB_SHIFT"] = _safe_number(emb)

    # ---------- Explainability proxy drift ----------
    explain_results = feature_importance_shift(model, ref_X.columns)
    # feature_importance_shift likely returns dict of metrics; flatten
    if isinstance(explain_results, dict):
        for k, v in explain_results.items():
            results[f"EXPL_{k}"] = _safe_number(v)
    else:
        results["EXPL_RESULT"] = json.dumps(explain_results, ensure_ascii=False)

    # ---------- Prediction-space drift ----------
    results["PRED_mean_shift"] = _safe_number(prediction_mean_shift(model, ref_X, cur_X))
    results["PRED_distribution_shift"] = _safe_number(prediction_distribution_shift(model, ref_X, cur_X))
    results["PRED_confidence_shift"] = _safe_number(prediction_confidence_shift(model, ref_X, cur_X))

    # Helpful meta info
    results["TOTAL_FEATURES_CHECKED"] = _safe_number(len(ref_X.columns))
    results["NUM_STAT_TESTS"] = _safe_number(len(ref_X.columns) * 2)  # KS + WASS per feature

    return results
