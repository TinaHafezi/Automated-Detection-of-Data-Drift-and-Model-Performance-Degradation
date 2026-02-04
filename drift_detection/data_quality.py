import pandas as pd

def check_missing_drift(ref_df, cur_df):
    ref_missing = ref_df.isna().mean()
    cur_missing = cur_df.isna().mean()
    drift = (cur_missing - ref_missing).abs()
    return drift.to_dict()

def check_zero_drift(ref_df, cur_df):
    ref_zero = (ref_df == 0).mean()
    cur_zero = (cur_df == 0).mean()
    drift = (cur_zero - ref_zero).abs()
    return drift.to_dict()

def detect_new_categories(ref_df, cur_df):
    new_cat = {}
    for col in ref_df.select_dtypes(include="object").columns:
        ref_vals = set(ref_df[col].unique())
        cur_vals = set(cur_df[col].unique())
        new_cat[col] = list(cur_vals - ref_vals)
    return new_cat
