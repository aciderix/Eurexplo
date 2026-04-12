"""
Eurexplo — Advanced exploration: wavelets, compression, graphs, embeddings,
spectral analysis, transition matrices, information theory.
"""
import numpy as np
import json
from collections import Counter
from itertools import combinations

data = np.load('/home/user/Eurexplo/euromillions_data.npz')
numbers = data['numbers']
stars = data['stars']
has_winner = data['has_winner']
jackpots = data['jackpots']
with open('/home/user/Eurexplo/dates.json') as f:
    dates = json.load(f)

N = len(numbers)
star1 = stars[:, 0]
star2 = stars[:, 1]

# ============================================================
# 2. WAVELETS ON STAR2
# ============================================================
print("="*60)
print("2. WAVELET ANALYSIS ON STAR2")
print("="*60)

# Manual Haar wavelet decomposition (no pywt dependency)
def haar_decompose(signal, levels=5):
    """Haar wavelet decomposition — returns detail coefficients at each level."""
    details = []
    approx = signal.astype(float).copy()
    for level in range(levels):
        n = len(approx)
        if n < 2:
            break
        n_even = n - (n % 2)
        a = approx[:n_even].reshape(-1, 2)
        new_approx = a.mean(axis=1)
        detail = a[:, 0] - a[:, 1]
        details.append(detail)
        approx = new_approx
    return details, approx

details, approx = haar_decompose(star2, levels=8)
print("\nHaar wavelet detail energy by level:")
for i, d in enumerate(details):
    energy = np.sum(d**2)
    mean_abs = np.mean(np.abs(d))
    print(f"  Level {i+1} (scale ~{2**(i+1)} draws): energy={energy:.1f}, mean|d|={mean_abs:.3f}, len={len(d)}")

# Compare to null model
print("\nNull model comparison (1000 permutations):")
for level_check in [0, 2, 4]:
    real_energy = np.sum(details[level_check]**2)
    null_energies = []
    for _ in range(1000):
        shuf = np.random.permutation(star2)
        d_null, _ = haar_decompose(shuf, levels=level_check+1)
        null_energies.append(np.sum(d_null[level_check]**2))
    null_energies = np.array(null_energies)
    z = (real_energy - null_energies.mean()) / null_energies.std()
    print(f"  Level {level_check+1}: real={real_energy:.1f}, null={null_energies.mean():.1f}±{null_energies.std():.1f}, z={z:+.2f}σ")

# ============================================================
# 3. COMPRESSION (KOLMOGOROV PROXY)
# ============================================================
print("\n" + "="*60)
print("3. COMPRESSION ANALYSIS (Kolmogorov proxy)")
print("="*60)

import zlib

def compression_ratio(arr):
    """Compress array to bytes and return ratio."""
    raw = arr.tobytes()
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / len(raw)

# Numbers
cr_nums = compression_ratio(numbers)
print(f"\nNumbers compression ratio: {cr_nums:.4f}")

# Stars
cr_stars = compression_ratio(stars)
print(f"Stars compression ratio: {cr_stars:.4f}")

# Star2 alone
cr_s2 = compression_ratio(star2)
print(f"Star2 compression ratio: {cr_s2:.4f}")

# Null model comparison
print("\nNull model (1000 random):")
null_cr_nums = []
null_cr_stars = []
null_cr_s2 = []
for _ in range(1000):
    rand_nums = np.sort(np.array([np.random.choice(range(1,51), 5, replace=False) for _ in range(N)]), axis=1).astype(np.int8)
    rand_stars = np.sort(np.array([np.random.choice(range(1,13), 2, replace=False) for _ in range(N)]), axis=1).astype(np.int8)
    null_cr_nums.append(compression_ratio(rand_nums))
    null_cr_stars.append(compression_ratio(rand_stars))
    null_cr_s2.append(compression_ratio(rand_stars[:, 1]))

null_cr_nums = np.array(null_cr_nums)
null_cr_stars = np.array(null_cr_stars)
null_cr_s2 = np.array(null_cr_s2)

z_nums = (cr_nums - null_cr_nums.mean()) / null_cr_nums.std()
z_stars = (cr_stars - null_cr_stars.mean()) / null_cr_stars.std()
z_s2 = (cr_s2 - null_cr_s2.mean()) / null_cr_s2.std()

print(f"  Numbers: real={cr_nums:.4f}, null={null_cr_nums.mean():.4f}±{null_cr_nums.std():.4f}, z={z_nums:+.2f}σ")
print(f"  Stars:   real={cr_stars:.4f}, null={null_cr_stars.mean():.4f}±{null_cr_stars.std():.4f}, z={z_stars:+.2f}σ")
print(f"  Star2:   real={cr_s2:.4f}, null={null_cr_s2.mean():.4f}±{null_cr_s2.std():.4f}, z={z_s2:+.2f}σ")

# Sliding window compression — detect regime changes
print("\nSliding window compression (window=200, step=50) on star2:")
window_c = 200
step_c = 50
for start in range(0, N - window_c, step_c):
    end = start + window_c
    cr = compression_ratio(star2[start:end])
    year_start = dates[start][:4]
    year_end = dates[min(end, N-1)][:4]
    print(f"  [{start:4d}-{end:4d}] ({year_start}-{year_end}): ratio={cr:.4f}")

# ============================================================
# 4. GRAPH ANALYSIS ON NUMBERS
# ============================================================
print("\n" + "="*60)
print("4. GRAPH ANALYSIS — CO-OCCURRENCE NETWORK")
print("="*60)

# Build adjacency matrix: edge weight = co-occurrence count
adj = np.zeros((50, 50), dtype=int)
for draw in numbers:
    for i, j in combinations(range(5), 2):
        a, b = draw[i] - 1, draw[j] - 1
        adj[a][b] += 1
        adj[b][a] += 1

# Expected co-occurrence under uniform model
# P(both i,j in draw) = C(48,3)/C(50,5) for i≠j
from math import comb
p_pair = comb(48, 3) / comb(50, 5)
expected_cooc = N * p_pair
print(f"Expected co-occurrence per pair: {expected_cooc:.2f}")

# Find most and least co-occurring pairs
pair_list = []
for i in range(50):
    for j in range(i+1, 50):
        pair_list.append((adj[i][j], i+1, j+1))
pair_list.sort(reverse=True)

print(f"\nTop 10 most co-occurring pairs:")
for count, a, b in pair_list[:10]:
    z = (count - expected_cooc) / np.sqrt(expected_cooc)
    print(f"  ({a:2d}, {b:2d}): {count} (z={z:+.2f})")

print(f"\nBottom 10 least co-occurring pairs:")
for count, a, b in pair_list[-10:]:
    z = (count - expected_cooc) / np.sqrt(expected_cooc)
    print(f"  ({a:2d}, {b:2d}): {count} (z={z:+.2f})")

# Degree distribution (sum of co-occurrences per number)
degrees = adj.sum(axis=1)
degree_mean = degrees.mean()
degree_std = degrees.std()
print(f"\nDegree stats: mean={degree_mean:.1f}, std={degree_std:.1f}")
outliers = np.where(np.abs(degrees - degree_mean) > 2 * degree_std)[0]
if len(outliers) > 0:
    print(f"Degree outliers (>2σ): {[o+1 for o in outliers]}")
else:
    print("No degree outliers (>2σ)")

# Spectral analysis of adjacency matrix
eigenvalues = np.linalg.eigvalsh(adj.astype(float))
eigenvalues = np.sort(eigenvalues)[::-1]
print(f"\nTop 5 eigenvalues of adjacency matrix: {eigenvalues[:5].round(2)}")
print(f"Spectral gap (λ1-λ2): {eigenvalues[0]-eigenvalues[1]:.2f}")

# Compare spectral gap to null model
print("\nSpectral gap null model (200 random graphs):")
null_gaps = []
for _ in range(200):
    rand_adj = np.zeros((50, 50))
    for _ in range(N):
        draw = np.sort(np.random.choice(range(50), 5, replace=False))
        for i, j in combinations(range(5), 2):
            rand_adj[draw[i]][draw[j]] += 1
            rand_adj[draw[j]][draw[i]] += 1
    eigs = np.linalg.eigvalsh(rand_adj)
    eigs = np.sort(eigs)[::-1]
    null_gaps.append(eigs[0] - eigs[1])
null_gaps = np.array(null_gaps)
real_gap = eigenvalues[0] - eigenvalues[1]
z_gap = (real_gap - null_gaps.mean()) / null_gaps.std()
print(f"  Real gap={real_gap:.2f}, null={null_gaps.mean():.2f}±{null_gaps.std():.2f}, z={z_gap:+.2f}σ")

# ============================================================
# 5. TRANSITION MATRIX ON STARS
# ============================================================
print("\n" + "="*60)
print("5. TRANSITION MATRIX — STAR PAIRS")
print("="*60)

# Star pair transitions: (s1,s2) at time t -> (s1,s2) at time t+1
star_pairs = list(zip(star1, star2))
trans_count = {}
for i in range(len(star_pairs) - 1):
    key = (tuple(star_pairs[i]), tuple(star_pairs[i+1]))
    trans_count[key] = trans_count.get(key, 0) + 1

# Most common transitions
trans_sorted = sorted(trans_count.items(), key=lambda x: -x[1])
print("Top 15 most frequent transitions (s1,s2 -> s1,s2):")
for (fr, to), count in trans_sorted[:15]:
    print(f"  {fr} -> {to}: {count}")

# Star2 transition matrix
print("\nStar2 transition matrix analysis:")
trans_s2 = np.zeros((12, 12))
for i in range(N-1):
    trans_s2[star2[i]-1][star2[i+1]-1] += 1

# Normalize rows
row_sums = trans_s2.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
trans_s2_prob = trans_s2 / row_sums

# Find strongest transitions
print("Strongest star2 transitions (P > 0.15):")
for i in range(12):
    for j in range(12):
        if trans_s2_prob[i][j] > 0.15:
            expected_p = row_sums[j].item() / (N-1)  # marginal probability
            print(f"  {i+1} -> {j+1}: P={trans_s2_prob[i][j]:.3f} (marginal={expected_p:.3f})")

# Stationary distribution of transition matrix
eigvals, eigvecs = np.linalg.eig(trans_s2_prob.T)
# Find eigenvector for eigenvalue closest to 1
idx = np.argmin(np.abs(eigvals - 1))
stationary = np.abs(eigvecs[:, idx])
stationary = stationary / stationary.sum()
print(f"\nStationary distribution of star2 Markov chain:")
for i in range(12):
    empirical = (star2 == i+1).sum() / N
    print(f"  Star {i+1:2d}: stationary={stationary[i]:.4f}, empirical={empirical:.4f}")

# ============================================================
# 6. INFORMATION THEORY — ENTROPY
# ============================================================
print("\n" + "="*60)
print("6. INFORMATION THEORY")
print("="*60)

def entropy(counts):
    """Shannon entropy from counts."""
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# Overall entropy
star2_counts = np.bincount(star2, minlength=13)[1:]
h_star2 = entropy(star2_counts)
h_max = np.log2(12)
print(f"\nStar2 entropy: {h_star2:.4f} bits (max={h_max:.4f}, efficiency={h_star2/h_max:.4f})")

star1_counts = np.bincount(star1, minlength=13)[1:]
h_star1 = entropy(star1_counts)
print(f"Star1 entropy: {h_star1:.4f} bits (max={h_max:.4f}, efficiency={h_star1/h_max:.4f})")

# Conditional entropy H(star2_t | star2_{t-1})
h_cond = 0
for i in range(12):
    row = trans_s2[i]
    if row.sum() > 0:
        h_row = entropy(row)
        h_cond += (row.sum() / (N-1)) * h_row
print(f"\nConditional entropy H(star2_t | star2_{{t-1}}): {h_cond:.4f} bits")
print(f"Mutual information I(star2_t ; star2_{{t-1}}): {h_star2 - h_cond:.4f} bits")

# Sliding window entropy for star2
print("\nSliding window entropy (star2, window=100, step=50):")
for start in range(0, N - 100, 100):
    end = start + 100
    window_counts = np.bincount(star2[start:end], minlength=13)[1:]
    h_w = entropy(window_counts)
    year = dates[start][:4]
    print(f"  [{start:4d}-{end:4d}] ({year}): H={h_w:.4f} bits ({h_w/h_max:.3f} eff)")

# ============================================================
# 7. NUMBERS — DEEPER STRUCTURAL ANALYSIS
# ============================================================
print("\n" + "="*60)
print("7. NUMBERS — DEEPER STRUCTURAL ANALYSIS")
print("="*60)

# 7.1 Sum series analysis
sums = numbers.sum(axis=1)
print(f"\n--- 7.1 Sum of numbers ---")
print(f"Mean={sums.mean():.2f} (theoretical=127.5), Std={sums.std():.2f}")

# Autocorrelation of sums
for lag in [1, 2, 3, 5, 10]:
    r = np.corrcoef(sums[:-lag], sums[lag:])[0, 1]
    print(f"  Sum autocorr lag {lag:2d}: r={r:+.4f}")

# 7.2 Spread (max - min)
spreads = numbers[:, 4] - numbers[:, 0]
print(f"\n--- 7.2 Spread (max-min) ---")
print(f"Mean={spreads.mean():.2f}, Std={spreads.std():.2f}")
for lag in [1, 2, 3]:
    r = np.corrcoef(spreads[:-lag], spreads[lag:])[0, 1]
    print(f"  Spread autocorr lag {lag}: r={r:+.4f}")

# 7.3 Gap analysis — differences between consecutive sorted numbers
print(f"\n--- 7.3 Gap pattern ---")
gaps = np.diff(numbers, axis=1)  # (N, 4) differences
gap_means = gaps.mean(axis=0)
gap_stds = gaps.std(axis=0)
print(f"Mean gaps: {gap_means.round(2)}")
print(f"Std gaps:  {gap_stds.round(2)}")

# Autocorrelation of gap patterns
for pos in range(4):
    r = np.corrcoef(gaps[:-1, pos], gaps[1:, pos])[0, 1]
    print(f"  Gap[{pos}] autocorr lag 1: r={r:+.4f}")

# 7.4 Even/odd pattern
even_counts = np.sum(numbers % 2 == 0, axis=1)
print(f"\n--- 7.4 Even count per draw ---")
even_dist = np.bincount(even_counts, minlength=6)
print(f"Distribution: {dict(zip(range(6), even_dist))}")
# Autocorrelation
r = np.corrcoef(even_counts[:-1], even_counts[1:])[0, 1]
print(f"Even count autocorr lag 1: r={r:+.4f}")

# 7.5 Decade distribution per draw
print(f"\n--- 7.5 Decade distribution ---")
decades = (numbers - 1) // 10  # 0-4
for d in range(5):
    counts_per_draw = (decades == d).sum(axis=1)
    print(f"  Decade {d*10+1}-{(d+1)*10}: mean={counts_per_draw.mean():.3f}")

# 7.6 Numbers as 5D vectors — PCA
print(f"\n--- 7.6 PCA on number vectors ---")
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
nums_centered = numbers.astype(float) - numbers.mean(axis=0)
pca.fit(nums_centered)
print(f"Explained variance ratios: {pca.explained_variance_ratio_.round(4)}")
print(f"Singular values: {pca.singular_values_.round(2)}")

# Compare to null model
null_evr = []
for _ in range(500):
    rand_nums = np.sort(np.array([np.random.choice(range(1,51), 5, replace=False) for _ in range(N)]), axis=1).astype(float)
    rand_centered = rand_nums - rand_nums.mean(axis=0)
    pca_null = PCA(n_components=5)
    pca_null.fit(rand_centered)
    null_evr.append(pca_null.explained_variance_ratio_)
null_evr = np.array(null_evr)
print(f"Null model EVR mean: {null_evr.mean(axis=0).round(4)}")
for i in range(5):
    z = (pca.explained_variance_ratio_[i] - null_evr[:, i].mean()) / null_evr[:, i].std()
    print(f"  PC{i+1}: z={z:+.2f}σ")

# ============================================================
# 8. STAR PAIR ANALYSIS — JOINT DISTRIBUTION
# ============================================================
print("\n" + "="*60)
print("8. STAR PAIR JOINT DISTRIBUTION")
print("="*60)

# Joint distribution of (star1, star2)
joint = np.zeros((12, 12), dtype=int)
for s1, s2 in stars:
    joint[s1-1][s2-1] += 1

# Under independence: P(s1,s2) = P(s1)*P(s2)
s1_probs = star1_counts / star1_counts.sum()
s2_probs = star2_counts / star2_counts.sum()
expected_joint = np.outer(s1_probs, s2_probs) * N

# Chi2 for independence
chi2_joint = np.sum((joint - expected_joint)**2 / np.maximum(expected_joint, 0.001))
# df = (12-1)*(12-1) = 121, but many cells have constraint s1 < s2
# Actually, joint[i][j] only valid for j > i (since star1 < star2)
print(f"Joint distribution chi2 (independence test): {chi2_joint:.1f}")

# Filter for valid pairs (s1 < s2) only
valid_count = 0
chi2_valid = 0
for i in range(12):
    for j in range(i+1, 12):
        obs = joint[i][j]
        # Expected: P(pair (i+1,j+1)) = 2 * P(draw i+1) * P(draw j+1) [order doesn't matter]
        p_pair_ij = 2 * (star1_counts[i] + star2_counts[i]) / (2*N) * (star1_counts[j] + star2_counts[j]) / (2*N)
        exp_ij = N * p_pair_ij
        if exp_ij > 0:
            chi2_valid += (obs - exp_ij)**2 / exp_ij
            valid_count += 1

# Most over/under-represented pairs
print(f"\nMost over-represented star pairs:")
pair_devs = []
for i in range(12):
    for j in range(i+1, 12):
        obs = joint[i][j]
        exp = expected_joint[i][j]
        if exp > 1:
            z = (obs - exp) / np.sqrt(exp)
            pair_devs.append((z, i+1, j+1, obs, exp))
pair_devs.sort(reverse=True)
for z, s1, s2, obs, exp in pair_devs[:10]:
    print(f"  ({s1:2d},{s2:2d}): obs={obs:3d}, exp={exp:.1f}, z={z:+.2f}")
print(f"\nMost under-represented:")
for z, s1, s2, obs, exp in pair_devs[-5:]:
    print(f"  ({s1:2d},{s2:2d}): obs={obs:3d}, exp={exp:.1f}, z={z:+.2f}")

# ============================================================
# 9. TEMPORAL SEGMENTATION — PRE/POST 2016
# ============================================================
print("\n" + "="*60)
print("9. TEMPORAL SEGMENTATION")
print("="*60)

# Find index where 2016 starts
idx_2016 = next(i for i, d in enumerate(dates) if d >= '2016-09-27')
print(f"Split at draw #{idx_2016} ({dates[idx_2016]})")

for period_name, sl in [("Pre-2016", slice(0, idx_2016)), ("Post-2016", slice(idx_2016, N))]:
    period_s2 = star2[sl]
    n_period = len(period_s2)
    counts = np.bincount(period_s2, minlength=13)[1:]
    expected_p = n_period / 12
    chi2_p = np.sum((counts - expected_p)**2 / expected_p)
    h = entropy(counts)

    print(f"\n{period_name} (n={n_period}):")
    print(f"  Chi2 = {chi2_p:.1f} (critical = 19.7)")
    print(f"  Entropy = {h:.4f} (max = {np.log2(12):.4f})")
    print(f"  Top 3 stars: {[c[0] for c in Counter(period_s2.tolist()).most_common(3)]}")

    # Autocorrelation star2
    for lag in [1, 3, 6]:
        if n_period > lag:
            r = np.corrcoef(period_s2[:-lag], period_s2[lag:])[0, 1]
            print(f"  Autocorr lag {lag}: r={r:+.4f}")

    # Rolling mode predictor
    w = 50 if period_name == "Pre-2016" else 100
    correct = 0
    total_p = 0
    for i in range(w, n_period):
        recent = period_s2[i-w:i]
        mode = Counter(recent.tolist()).most_common(1)[0][0]
        if period_s2[i] == mode:
            correct += 1
        total_p += 1
    if total_p > 0:
        acc = correct / total_p
        z = (acc - 1/12) / np.sqrt((1/12)*(11/12)/total_p)
        print(f"  Rolling mode (w={w}): {100*acc:.2f}% (z={z:+.1f}σ)")

# ============================================================
# 10. NUMBER-STAR CROSS CORRELATION
# ============================================================
print("\n" + "="*60)
print("10. NUMBER-STAR CROSS CORRELATION")
print("="*60)

# Does the sum of numbers predict stars?
for target_name, target in [("star1", star1), ("star2", star2)]:
    r_sum = np.corrcoef(sums, target)[0, 1]
    r_spread = np.corrcoef(spreads, target)[0, 1]
    r_even = np.corrcoef(even_counts, target)[0, 1]
    print(f"{target_name}: corr w/ sum={r_sum:+.4f}, spread={r_spread:+.4f}, even_count={r_even:+.4f}")

# Lagged cross-correlation
print("\nLagged cross-correlations (sums -> star2):")
for lag in range(1, 6):
    r = np.corrcoef(sums[:-lag], star2[lag:])[0, 1]
    print(f"  sum(t) -> star2(t+{lag}): r={r:+.4f}")

print("\nLagged cross-correlations (star2 -> sums):")
for lag in range(1, 6):
    r = np.corrcoef(star2[:-lag], sums[lag:])[0, 1]
    print(f"  star2(t) -> sum(t+{lag}): r={r:+.4f}")

print("\n" + "="*60)
print("ADVANCED EXPLORATION COMPLETE")
print("="*60)
