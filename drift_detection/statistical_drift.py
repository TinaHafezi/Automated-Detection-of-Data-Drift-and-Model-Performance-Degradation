from scipy.stats import ks_2samp, chisquare, wasserstein_distance

def ks_drift(ref, cur):
    return ks_2samp(ref, cur).pvalue

def wasserstein_drift(ref, cur):
    return wasserstein_distance(ref, cur)

def chi2_drift(ref, cur):
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts().reindex(ref_counts.index, fill_value=0)
    return chisquare(cur_counts, ref_counts)[1]
