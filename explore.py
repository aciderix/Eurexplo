"""
Eurexplo — Full exploration engine.
Pipeline: generate -> measure -> detect patterns -> falsify -> loop
"""
import numpy as np
import json
from collections import Counter
from scipy import stats as sp_stats

# ============================================================
# LOAD DATA
# ============================================================
data = np.load('/home/user/Eurexplo/euromillions_data.npz')
numbers = data['numbers']  # (N, 5) int8
stars = data['stars']       # (N, 2) int8
has_winner = data['has_winner']
jackpots = data['jackpots']
with open('/home/user/Eurexplo/dates.json') as f:
    dates = json.load(f)

N = len(numbers)
print(f"=== EUREXPLO ENGINE ===")
print(f"Draws: {N}, Range: {dates[0]} -> {dates[-1]}")
print(f"Winners: {has_winner.sum()}/{N} ({100*has_winner.mean():.1f}%)")

star1 = stars[:, 0]
star2 = stars[:, 1]

# ============================================================
# PART 1: VERIFY PREVIOUS AGENT'S RESULTS
# ============================================================
print("\n" + "="*60)
print("PART 1: VERIFICATION OF PREVIOUS RESULTS")
print("="*60)

# 1.1 Numbers uniformity chi2
print("\n--- 1.1 Numbers Uniformity Chi2 ---")
num_flat = numbers.flatten()
num_counts = np.bincount(num_flat, minlength=51)[1:]  # 1-50
expected = N * 5 / 50
chi2_nums = np.sum((num_counts - expected)**2 / expected)
print(f"Chi2 = {chi2_nums:.1f} (df=49, critical α=0.05 = 67.5)")
print(f"Verdict: {'UNIFORM' if chi2_nums < 67.5 else 'NOT UNIFORM'}")

# 1.2 Stars uniformity chi2
print("\n--- 1.2 Stars Uniformity Chi2 ---")
star_flat = stars.flatten()
star_counts = np.bincount(star_flat, minlength=13)[1:]  # 1-12
expected_s = N * 2 / 12
chi2_stars = np.sum((star_counts - expected_s)**2 / expected_s)
print(f"Chi2 = {chi2_stars:.1f} (df=11, critical α=0.05 = 19.7)")
print(f"Verdict: {'UNIFORM' if chi2_stars < 19.7 else 'NOT UNIFORM'}")
print(f"\nStar frequencies:")
for i in range(12):
    z = (star_counts[i] - expected_s) / np.sqrt(expected_s)
    flag = "***" if abs(z) > 2 else ""
    print(f"  Star {i+1:2d}: {star_counts[i]:4d} (expected {expected_s:.1f}, z={z:+.2f}) {flag}")

# 1.3 Star2 autocorrelation
print("\n--- 1.3 Star2 Autocorrelation ---")
s2_centered = star2 - star2.mean()
s2_var = np.var(star2)
for lag in range(1, 11):
    r = np.corrcoef(star2[:-lag], star2[lag:])[0, 1]
    # Null model: permutation test
    n_perm = 5000
    perm_rs = np.zeros(n_perm)
    for p in range(n_perm):
        shuffled = np.random.permutation(star2)
        perm_rs[p] = np.corrcoef(shuffled[:-lag], shuffled[lag:])[0, 1]
    z = (r - perm_rs.mean()) / perm_rs.std()
    sig = "***" if abs(z) > 3 else ""
    print(f"  lag {lag:2d}: r={r:+.4f}, z={z:+.2f}σ {sig}")

# 1.4 Star1 autocorrelation (should be null)
print("\n--- 1.4 Star1 Autocorrelation ---")
for lag in [1, 2, 3, 5]:
    r = np.corrcoef(star1[:-lag], star1[lag:])[0, 1]
    print(f"  lag {lag:2d}: r={r:+.4f}")

# 1.5 Regime change star12
print("\n--- 1.5 Regime Change: Star 12 ---")
s12_mask = (stars == 12).any(axis=1)
first_s12 = np.where(s12_mask)[0]
if len(first_s12) > 0:
    idx = first_s12[0]
    print(f"First appearance of star 12: draw #{idx} ({dates[idx]})")
    print(f"Before: {s12_mask[:idx].sum()} appearances in {idx} draws")
    print(f"After: {s12_mask[idx:].sum()} appearances in {N-idx} draws")
else:
    print("Star 12 never appeared!")

# 1.6 Rolling mode predictor verification
print("\n--- 1.6 Rolling Mode Predictor (star2, w=100) ---")
window = 100
correct_1 = 0
correct_top3 = 0
total = 0
for i in range(window, N):
    recent = star2[i-window:i]
    counts = Counter(recent.tolist())
    mode_val = counts.most_common(1)[0][0]
    top3 = [c[0] for c in counts.most_common(3)]
    actual = star2[i]
    if actual == mode_val:
        correct_1 += 1
    if actual in top3:
        correct_top3 += 1
    total += 1

acc_1 = correct_1 / total
acc_3 = correct_top3 / total
baseline_1 = 1/12
baseline_3 = 3/12
z_1 = (acc_1 - baseline_1) / np.sqrt(baseline_1 * (1 - baseline_1) / total)
z_3 = (acc_3 - baseline_3) / np.sqrt(baseline_3 * (1 - baseline_3) / total)
print(f"Mode accuracy: {100*acc_1:.2f}% (baseline {100*baseline_1:.2f}%, z={z_1:+.1f}σ)")
print(f"Top-3 accuracy: {100*acc_3:.2f}% (baseline {100*baseline_3:.2f}%, z={z_3:+.1f}σ)")

# OOS on last 300
print("\n--- 1.7 OOS Validation (last 300) ---")
correct_oos = 0
total_oos = 0
for i in range(N-300, N):
    recent = star2[i-window:i]
    counts = Counter(recent.tolist())
    mode_val = counts.most_common(1)[0][0]
    if star2[i] == mode_val:
        correct_oos += 1
    total_oos += 1
acc_oos = correct_oos / total_oos
z_oos = (acc_oos - baseline_1) / np.sqrt(baseline_1 * (1 - baseline_1) / total_oos)
print(f"OOS Mode accuracy: {100*acc_oos:.2f}% (z={z_oos:+.1f}σ)")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
