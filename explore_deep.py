"""
Eurexplo — Deep dive into partially confirmed signals + new angles.
"""
import numpy as np
import json
from collections import Counter
from datetime import datetime
from scipy import stats

data = np.load('/home/user/Eurexplo/euromillions_data.npz')
numbers = data['numbers']
stars = data['stars']
with open('/home/user/Eurexplo/dates.json') as f:
    dates = json.load(f)

N = len(numbers)
star1 = stars[:, 0]
star2 = stars[:, 1]

# ============================================================
# D1. DAY-OF-WEEK DEEP DIVE
# ============================================================
print("="*70)
print("D1. DAY-OF-WEEK — DEEP DIVE")
print("="*70)

days = np.array([datetime.strptime(d, '%Y-%m-%d').weekday() for d in dates])
tue_mask = days == 1
fri_mask = days == 4

# Key question: when did Tuesday draws start?
print("\nDraw days per year:")
for year in range(2004, 2027):
    year_mask = np.array([d.startswith(str(year)) for d in dates])
    if year_mask.sum() == 0:
        continue
    n_tue = (days[year_mask] == 1).sum()
    n_fri = (days[year_mask] == 4).sum()
    if n_tue > 0 or n_fri > 0:
        print(f"  {year}: Tue={n_tue}, Fri={n_fri}, Total={year_mask.sum()}")

# Is the Tue/Fri difference explained by temporal location?
# Pre-2011: only Friday draws. Post-2011: Tue+Fri
idx_2011 = next(i for i, d in enumerate(dates) if d >= '2011-01-01')
print(f"\nPre-2011 (only Fri, n={idx_2011}): star2 mean = {star2[:idx_2011].mean():.3f}")
print(f"Post-2011 (Tue+Fri, n={N-idx_2011}): star2 mean = {star2[idx_2011:].mean():.3f}")

# Within post-2011, compare Tue vs Fri
post_2011_mask = np.arange(N) >= idx_2011
tue_post = star2[post_2011_mask & tue_mask]
fri_post = star2[post_2011_mask & fri_mask]
t, p = stats.ttest_ind(tue_post, fri_post)
print(f"\nPost-2011 only:")
print(f"  Tue: mean={tue_post.mean():.3f}, n={len(tue_post)}")
print(f"  Fri: mean={fri_post.mean():.3f}, n={len(fri_post)}")
print(f"  t={t:.3f}, p={p:.4f}")

# Check star2 distributions by day
print("\nStar2 distributions (post-2011):")
for day, label in [(1, "Tue"), (4, "Fri")]:
    mask = post_2011_mask & (days == day)
    s2d = star2[mask]
    counts = np.bincount(s2d, minlength=13)[1:]
    total = counts.sum()
    top3 = Counter(s2d.tolist()).most_common(3)
    print(f"  {label}: top3={top3}, entropy={-np.sum((counts/total)*np.log2(counts/total+1e-10)):.3f}")

# Rolling mode per day
print("\nRolling mode per day (post-2011):")
for day, label in [(1, "Tue"), (4, "Fri")]:
    mask = np.where(post_2011_mask & (days == day))[0]
    if len(mask) < 50:
        continue
    correct = 0
    total = 0
    for idx in range(50, len(mask)):
        recent = star2[mask[idx-50:idx]]
        mode = Counter(recent.tolist()).most_common(1)[0][0]
        if star2[mask[idx]] == mode:
            correct += 1
        total += 1
    acc = correct / total
    z = (acc - 1/12) / np.sqrt((1/12)*(11/12)/total)
    print(f"  {label}: {100*acc:.2f}% (n={total}, z={z:+.1f}σ)")

# ============================================================
# D2. STAR1 — HIDDEN SIGNAL?
# ============================================================
print("\n" + "="*70)
print("D2. STAR1 — HIDDEN SIGNAL SEARCH")
print("="*70)

# Star1 = min(star1, star2). Different distribution.
s1_counts = np.bincount(star1, minlength=13)[1:]
expected_s1 = N / 12
chi2_s1 = np.sum((s1_counts - expected_s1)**2 / expected_s1)
print(f"Star1 chi2: {chi2_s1:.1f} (critical = 19.7)")
print(f"Star1 frequencies:")
for i in range(12):
    z = (s1_counts[i] - expected_s1) / np.sqrt(expected_s1)
    flag = "***" if abs(z) > 2 else ""
    print(f"  Star {i+1:2d}: {s1_counts[i]:4d} (z={z:+.2f}) {flag}")

# Rolling mode on star1
correct_s1 = 0
for i in range(100, N):
    recent = star1[i-100:i]
    mode = Counter(recent.tolist()).most_common(1)[0][0]
    if star1[i] == mode:
        correct_s1 += 1
acc_s1 = correct_s1 / (N - 100)
z_s1 = (acc_s1 - 1/12) / np.sqrt((1/12)*(11/12)/(N-100))
print(f"\nRolling mode star1 (w=100): {100*acc_s1:.2f}% (z={z_s1:+.1f}σ)")

# Combined star prediction: rolling mode for both
correct_both = 0
for i in range(100, N):
    pred_s1 = Counter(star1[i-100:i].tolist()).most_common(1)[0][0]
    pred_s2 = Counter(star2[i-100:i].tolist()).most_common(1)[0][0]
    if star1[i] == pred_s1 and star2[i] == pred_s2:
        correct_both += 1
acc_both = correct_both / (N - 100)
baseline_both = (1/12)**2
z_both = (acc_both - baseline_both) / np.sqrt(baseline_both * (1-baseline_both)/(N-100))
print(f"Both stars correct: {100*acc_both:.2f}% (baseline {100*baseline_both:.2f}%, z={z_both:+.1f}σ)")

# Top-3 for both
correct_both3 = 0
for i in range(100, N):
    top3_s1 = [c[0] for c in Counter(star1[i-100:i].tolist()).most_common(3)]
    top3_s2 = [c[0] for c in Counter(star2[i-100:i].tolist()).most_common(3)]
    if star1[i] in top3_s1 and star2[i] in top3_s2:
        correct_both3 += 1
acc_both3 = correct_both3 / (N - 100)
baseline_both3 = (3/12)**2
z_both3 = (acc_both3 - baseline_both3) / np.sqrt(baseline_both3 * (1-baseline_both3)/(N-100))
print(f"Both stars in top-3: {100*acc_both3:.2f}% (baseline {100*baseline_both3:.2f}%, z={z_both3:+.1f}σ)")

# ============================================================
# D3. STAR SUM & DIFFERENCE — NEW FEATURES
# ============================================================
print("\n" + "="*70)
print("D3. STAR SUM AND DIFFERENCE ANALYSIS")
print("="*70)

star_sum = (star1 + star2).astype(int)
star_diff = (star2 - star1).astype(int)

print(f"Star sum: mean={star_sum.mean():.2f}, std={star_sum.std():.2f}")
print(f"Star diff: mean={star_diff.mean():.2f}, std={star_diff.std():.2f}")

# Autocorrelation of sum and diff
for name, series in [("sum", star_sum), ("diff", star_diff)]:
    print(f"\n{name} autocorrelations:")
    for lag in [1, 2, 3, 5]:
        r = np.corrcoef(series[:-lag], series[lag:])[0, 1]
        print(f"  lag {lag}: r={r:+.4f}")

# Rolling mode on star_sum
correct_sum = 0
for i in range(100, N):
    recent = star_sum[i-100:i]
    mode = Counter(recent.tolist()).most_common(1)[0][0]
    if star_sum[i] == mode:
        correct_sum += 1
acc_sum = correct_sum / (N - 100)
# Star sum ranges from 3 to 23, ~21 possible values
n_unique_sum = len(set(star_sum.tolist()))
baseline_sum = 1 / n_unique_sum
z_sum = (acc_sum - baseline_sum) / np.sqrt(baseline_sum * (1-baseline_sum)/(N-100))
print(f"\nRolling mode on star_sum: {100*acc_sum:.2f}% (baseline ~{100*baseline_sum:.2f}%, z={z_sum:+.1f}σ)")

# ============================================================
# D4. CONSECUTIVE DRAWS — REPEAT PATTERNS
# ============================================================
print("\n" + "="*70)
print("D4. REPEAT PATTERNS IN CONSECUTIVE DRAWS")
print("="*70)

# How many numbers repeat from draw t-1 to draw t?
repeats = np.zeros(N-1, dtype=int)
for i in range(1, N):
    s1 = set(numbers[i-1].tolist())
    s2 = set(numbers[i].tolist())
    repeats[i-1] = len(s1 & s2)

repeat_dist = np.bincount(repeats, minlength=6)
print("Numbers repeating from previous draw:")
for r in range(6):
    # Expected under independence: hypergeometric
    # P(k repeats) = C(5,k)*C(45,5-k)/C(50,5)
    from math import comb
    expected_p = comb(5, r) * comb(45, 5-r) / comb(50, 5)
    expected_n = expected_p * (N-1)
    print(f"  {r} repeats: obs={repeat_dist[r]}, expected={expected_n:.1f}")

# Does repeat count predict star2?
for n_rep in range(4):
    mask = repeats == n_rep
    if mask.sum() > 10:
        s2_vals = star2[1:][mask]
        print(f"  {n_rep} repeats -> star2 mean={s2_vals.mean():.2f} (n={mask.sum()})")

# Star repeats
star_repeats = np.zeros(N-1, dtype=int)
for i in range(1, N):
    s1 = set(stars[i-1].tolist())
    s2 = set(stars[i].tolist())
    star_repeats[i-1] = len(s1 & s2)

sr_dist = np.bincount(star_repeats, minlength=3)
print("\nStars repeating from previous draw:")
for r in range(3):
    expected_p = comb(2, r) * comb(10, 2-r) / comb(12, 2)
    expected_n = expected_p * (N-1)
    print(f"  {r} repeats: obs={sr_dist[r]}, expected={expected_n:.1f}")

# ============================================================
# D5. OPTIMIZED COMBINED PREDICTOR
# ============================================================
print("\n" + "="*70)
print("D5. OPTIMIZED COMBINED STAR PREDICTOR")
print("="*70)

# Strategy: use day-of-week aware rolling mode
# For Tuesday draws, use only previous Tuesday star2 values
# For Friday draws, use only previous Friday star2 values

print("\n--- Day-aware rolling mode (post-2011) ---")
correct_day_aware = 0
correct_day_naive = 0
total_da = 0

for i in range(idx_2011 + 100, N):
    day = days[i]
    # Day-aware: only use same day-of-week draws
    same_day_indices = [j for j in range(max(0, i-300), i) if days[j] == day]
    recent_same = [star2[j] for j in same_day_indices[-50:]]
    # Naive: use all recent draws
    recent_all = star2[i-100:i]

    if len(recent_same) >= 20:
        pred_aware = Counter(recent_same).most_common(1)[0][0]
        pred_naive = Counter(recent_all.tolist()).most_common(1)[0][0]

        if star2[i] == pred_aware:
            correct_day_aware += 1
        if star2[i] == pred_naive:
            correct_day_naive += 1
        total_da += 1

if total_da > 0:
    acc_aware = correct_day_aware / total_da
    acc_naive = correct_day_naive / total_da
    z_aware = (acc_aware - 1/12) / np.sqrt((1/12)*(11/12)/total_da)
    z_naive = (acc_naive - 1/12) / np.sqrt((1/12)*(11/12)/total_da)
    print(f"Day-aware:  {100*acc_aware:.2f}% (z={z_aware:+.1f}σ)")
    print(f"Day-naive:  {100*acc_naive:.2f}% (z={z_naive:+.1f}σ)")
    print(f"Difference: {100*(acc_aware - acc_naive):+.2f}pp")

# --- Window size optimization ---
print("\n--- Window size optimization (star2) ---")
for w in [20, 30, 50, 75, 100, 150, 200, 300]:
    if w >= N:
        continue
    correct = 0
    for i in range(w, N):
        recent = star2[i-w:i]
        mode = Counter(recent.tolist()).most_common(1)[0][0]
        if star2[i] == mode:
            correct += 1
    acc = correct / (N - w)
    z = (acc - 1/12) / np.sqrt((1/12)*(11/12)/(N-w))
    print(f"  w={w:3d}: {100*acc:.2f}% (z={z:+.1f}σ)")

# --- Top-K optimization ---
print("\n--- Top-K optimization (star2, w=100) ---")
for k in [1, 2, 3, 4, 5]:
    correct = 0
    for i in range(100, N):
        recent = star2[i-100:i]
        topk = [c[0] for c in Counter(recent.tolist()).most_common(k)]
        if star2[i] in topk:
            correct += 1
    acc = correct / (N - 100)
    baseline = k / 12
    z = (acc - baseline) / np.sqrt(baseline * (1-baseline)/(N-100))
    edge = acc / baseline
    print(f"  top-{k}: {100*acc:.2f}% (baseline {100*baseline:.2f}%, z={z:+.1f}σ, edge={edge:.2f}x)")

# --- Same analysis for star1 ---
print("\n--- Top-K optimization (star1, w=100) ---")
for k in [1, 2, 3, 4, 5]:
    correct = 0
    for i in range(100, N):
        recent = star1[i-100:i]
        topk = [c[0] for c in Counter(recent.tolist()).most_common(k)]
        if star1[i] in topk:
            correct += 1
    acc = correct / (N - 100)
    baseline = k / 12
    z = (acc - baseline) / np.sqrt(baseline * (1-baseline)/(N-100))
    edge = acc / baseline
    print(f"  top-{k}: {100*acc:.2f}% (baseline {100*baseline:.2f}%, z={z:+.1f}σ, edge={edge:.2f}x)")

# ============================================================
# D6. FINAL COMBINED STRATEGY WITH CONFIDENCE
# ============================================================
print("\n" + "="*70)
print("D6. BEST OVERALL STRATEGY — FINAL")
print("="*70)

# Best strategy: rolling mode top-3 for both star1 and star2
# Expected: if independent, P(both in top-3) = P(s1 in top3) * P(s2 in top3)
correct_combo = 0
total_combo = 0
for i in range(100, N):
    top3_s1 = [c[0] for c in Counter(star1[i-100:i].tolist()).most_common(3)]
    top3_s2 = [c[0] for c in Counter(star2[i-100:i].tolist()).most_common(3)]
    # Generate all 9 possible pairs
    hit_s1 = star1[i] in top3_s1
    hit_s2 = star2[i] in top3_s2
    if hit_s1 and hit_s2:
        correct_combo += 1
    total_combo += 1

acc_combo = correct_combo / total_combo
# Number of valid star pairs from top3 x top3
# Need s1 < s2, so out of 9 pairs, ~half are valid = C(6,2)-duplicates
# Actually: pairs where top3_s1[i] < top3_s2[j]
# Baseline: 9/C(12,2) = 9/66 ≈ 13.6%
baseline_combo = 9 / 66  # rough upper bound
print(f"Both stars in respective top-3: {100*acc_combo:.2f}%")
print(f"Baseline (if top-3 independent): ~{100*baseline_combo:.2f}%")

# More precise: what fraction of the 66 possible star pairs are covered?
covered_pairs = set()
for i in range(100, N):
    top3_s1 = [c[0] for c in Counter(star1[i-100:i].tolist()).most_common(3)]
    top3_s2 = [c[0] for c in Counter(star2[i-100:i].tolist()).most_common(3)]
    for a in top3_s1:
        for b in top3_s2:
            if a < b:
                covered_pairs.add((a, b))

# Typical number of valid pairs per draw
pair_counts = []
for i in range(100, N):
    top3_s1 = [c[0] for c in Counter(star1[i-100:i].tolist()).most_common(3)]
    top3_s2 = [c[0] for c in Counter(star2[i-100:i].tolist()).most_common(3)]
    valid = sum(1 for a in top3_s1 for b in top3_s2 if a < b)
    pair_counts.append(valid)
mean_pairs = np.mean(pair_counts)
baseline_pairs = mean_pairs / 66
z_combo = (acc_combo - baseline_pairs) / np.sqrt(baseline_pairs * (1 - baseline_pairs) / total_combo)
print(f"Average valid pairs per draw: {mean_pairs:.1f}/66")
print(f"Adjusted baseline: {100*baseline_pairs:.2f}%")
print(f"z-score: {z_combo:+.1f}σ")

# What about the EXACT star pair?
correct_exact = 0
for i in range(100, N):
    mode_s1 = Counter(star1[i-100:i].tolist()).most_common(1)[0][0]
    mode_s2 = Counter(star2[i-100:i].tolist()).most_common(1)[0][0]
    if star1[i] == mode_s1 and star2[i] == mode_s2:
        correct_exact += 1
acc_exact = correct_exact / (N - 100)
baseline_exact = 1/66
z_exact = (acc_exact - baseline_exact) / np.sqrt(baseline_exact * (1-baseline_exact)/(N-100))
print(f"\nExact star pair prediction: {100*acc_exact:.3f}% (baseline {100*baseline_exact:.3f}%, z={z_exact:+.1f}σ)")

print("\n" + "="*70)
print("DEEP EXPLORATION COMPLETE")
print("="*70)
