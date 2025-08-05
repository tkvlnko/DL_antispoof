import numpy as np

def compute_det_curve(target_scores, nontarget_scores):
    """
    Assumes bonaide = positive and spoof = negative
        frr: false rejection rate (miss rate)
        far: false acceptance rate
    """
    target_scores = np.atleast_1d(target_scores)
    nontarget_scores = np.atleast_1d(nontarget_scores)

    if target_scores.size == 0 or nontarget_scores.size == 0:
        raise ValueError("Both bonaide and spoof must be non-empty.")

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # ============ sort by score ascending;lower threshold = stricter
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # ============ Cumulative bonafide accepted and spoof accepted
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0.0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1.0), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 1e-6), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(bonafide_scores, other_scores):
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    fnr = 1.0 - (1.0 - frr)  # = frr
    abs_diffs = np.abs(far - frr)
    idx = np.argmin(abs_diffs)
    eer = (far[idx] + frr[idx]) / 2.0
    threshold_eer = thresholds[idx]
    return float(eer), float(threshold_eer)


def roc_curve(bonafide_scores, spoof_scores):
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    tpr = 1.0 - frr
    fpr = far
    return fpr, tpr, thresholds
