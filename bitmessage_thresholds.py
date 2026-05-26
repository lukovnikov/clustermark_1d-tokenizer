import numpy as np
from scipy.stats import binom

def compute_thresholds(n_bits=64, p_match=0.5, target_fprs=None, verbose=True):
    if target_fprs is None:
        target_fprs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    if verbose:
        print(f"Calculating thresholds for {n_bits}-bit message matching")
        print("=" * 50)

    thresholds = {}

    for fpr in target_fprs:
        for k in range(n_bits + 1):
            tail_prob = 1 - binom.cdf(k - 1, n_bits, p_match)
            if tail_prob <= fpr:
                threshold_bits = k
                break
        else:
            threshold_bits = n_bits + 1

        threshold_ratio = threshold_bits / n_bits
        actual_fpr = 1 - binom.cdf(threshold_bits - 1, n_bits, p_match)
        thresholds[fpr] = threshold_ratio

        if verbose:
            print(f"Target FPR: {fpr:8.7f}")
            print(f"Threshold bits: {threshold_bits:2.0f}")
            print(f"Threshold ratio: {threshold_ratio:.6f}")
            print(f"Actual FPR: {actual_fpr:.8f}")
            print("-" * 30)

    if verbose:
        print("\nSummary - Thresholds:")
        for fpr, threshold in thresholds.items():
            print(f"FPR {fpr:8.7f}: {threshold:.6f}")

    return thresholds

if __name__ == "__main__":
    thresholds = compute_thresholds(n_bits=64)
    print("Computed thresholds:", thresholds)
