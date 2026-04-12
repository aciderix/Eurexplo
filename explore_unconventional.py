"""
Eurexplo — Unconventional exploration:
- Cellular automata
- Dynamical systems (Lyapunov, recurrence)
- Symbolic regression / program synthesis
- Spectral clustering
- Higher-order Markov chains
- Star2 regime-aware predictor
- Numbers embedding via co-occurrence
"""
import numpy as np
import json
from collections import Counter
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import MDS

data = np.load('/home/user/Eurexplo/euromillions_data.npz')
numbers = data['numbers']
stars = data['stars']
with open('/home/user/Eurexplo/dates.json') as f:
    dates = json.load(f)

N = len(numbers)
star1 = stars[:, 0]
star2 = stars[:, 1]

# ============================================================
# 11. HIGHER-ORDER MARKOV ON STAR2
# ============================================================
print("="*60)
print("11. HIGHER-ORDER MARKOV CHAINS ON STAR2")
print("="*60)

# Order-2 Markov: P(star2_t | star2_{t-1}, star2_{t-2})
print("\n--- Order-2 Markov (bigram -> next) ---")
bigram_trans = {}
for i in range(2, N):
    key = (int(star2[i-2]), int(star2[i-1]))
    val = int(star2[i])
    if key not in bigram_trans:
        bigram_trans[key] = []
    bigram_trans[key].append(val)

# Predictive accuracy: given (t-2, t-1), predict mode
correct_o2 = 0
total_o2 = 0
for i in range(102, N):
    # Build empirical distribution from last 100
    key = (int(star2[i-2]), int(star2[i-1]))
    history = []
    for j in range(max(2, i-100), i):
        if (int(star2[j-2]), int(star2[j-1])) == key:
            history.append(int(star2[j]))
    if len(history) >= 3:
        pred = Counter(history).most_common(1)[0][0]
        if star2[i] == pred:
            correct_o2 += 1
        total_o2 += 1

if total_o2 > 0:
    acc_o2 = correct_o2 / total_o2
    z_o2 = (acc_o2 - 1/12) / np.sqrt((1/12)*(11/12)/total_o2)
    print(f"Order-2 Markov (mode): {100*acc_o2:.2f}% on {total_o2} predictions (z={z_o2:+.1f}σ)")

# Order-3 Markov
print("\n--- Order-3 Markov (trigram -> next) ---")
correct_o3 = 0
total_o3 = 0
for i in range(103, N):
    key = (int(star2[i-3]), int(star2[i-2]), int(star2[i-1]))
    history = []
    for j in range(max(3, i-200), i):
        if (int(star2[j-3]), int(star2[j-2]), int(star2[j-1])) == key:
            history.append(int(star2[j]))
    if len(history) >= 2:
        pred = Counter(history).most_common(1)[0][0]
        if star2[i] == pred:
            correct_o3 += 1
        total_o3 += 1

if total_o3 > 0:
    acc_o3 = correct_o3 / total_o3
    z_o3 = (acc_o3 - 1/12) / np.sqrt((1/12)*(11/12)/total_o3)
    print(f"Order-3 Markov (mode): {100*acc_o3:.2f}% on {total_o3} predictions (z={z_o3:+.1f}σ)")

# ============================================================
# 12. CELLULAR AUTOMATON ANALYSIS
# ============================================================
print("\n" + "="*60)
print("12. CELLULAR AUTOMATON PATTERNS")
print("="*60)

# Represent each draw as a 50-bit vector
binary_nums = np.zeros((N, 50), dtype=np.int8)
for i in range(N):
    for n in numbers[i]:
        binary_nums[i, n-1] = 1

# XOR between consecutive draws — does the "difference pattern" have structure?
xor_seq = np.logical_xor(binary_nums[:-1], binary_nums[1:]).astype(int)
xor_sums = xor_seq.sum(axis=1)  # Hamming distance between consecutive draws
print(f"\nHamming distance between consecutive draws:")
print(f"  Mean={xor_sums.mean():.2f} (theoretical for 2 random 5-of-50: ~9.0)")
print(f"  Std={xor_sums.std():.2f}")

# Theoretical: E[hamming] for 2 independent 5-of-50 draws
# = 2 * 5 * (1 - 5/50) = 2 * 5 * 0.9 = 9.0
# Each position: P(differ) = P(on in A) + P(on in B) - 2*P(on in both)
# = 2*(5/50) - 2*(5/50)^2 = 0.2 - 0.02 = 0.18
# E[hamming] = 50 * 0.18 = 9.0
print(f"  Theoretical mean: 9.0")

# Autocorrelation of hamming distances
for lag in [1, 2, 3]:
    r = np.corrcoef(xor_sums[:-lag], xor_sums[lag:])[0, 1]
    print(f"  Hamming dist autocorr lag {lag}: r={r:+.4f}")

# XOR pattern periodicity check
print(f"\nXOR column sums (which positions flip most often):")
col_sums = xor_seq.sum(axis=0)
top_flippers = np.argsort(col_sums)[::-1][:10]
bot_flippers = np.argsort(col_sums)[:10]
print(f"  Most flipping:  {[(f+1, col_sums[f]) for f in top_flippers]}")
print(f"  Least flipping: {[(f+1, col_sums[f]) for f in bot_flippers]}")

# ============================================================
# 13. RECURRENCE ANALYSIS (Dynamical Systems)
# ============================================================
print("\n" + "="*60)
print("13. RECURRENCE ANALYSIS")
print("="*60)

# Recurrence plot proxy: how often does star2 return to previous values?
print("\n--- Star2 return times ---")
last_seen = {}
return_times = []
for i, s in enumerate(star2):
    s = int(s)
    if s in last_seen:
        return_times.append(i - last_seen[s])
    last_seen[s] = i

rt = np.array(return_times)
print(f"Mean return time: {rt.mean():.2f} (theoretical for uniform 1/12: ~6)")
print(f"Std return time: {rt.std():.2f}")
print(f"Return time distribution:")
rt_counts = np.bincount(rt, minlength=25)[:25]
for t in range(1, 25):
    expected = (11/12)**(t-1) * (1/12) * len(return_times)  # geometric
    print(f"  t={t:2d}: obs={rt_counts[t]:4d}, expected={expected:.0f}")

# Per-value return times
print("\nMean return time per star2 value:")
for val in range(1, 13):
    indices = np.where(star2 == val)[0]
    if len(indices) > 1:
        rts = np.diff(indices)
        print(f"  Star {val:2d}: mean={rts.mean():.1f}, min={rts.min()}, max={rts.max()}, n={len(rts)}")

# ============================================================
# 14. SPECTRAL CLUSTERING ON DRAWS
# ============================================================
print("\n" + "="*60)
print("14. SPECTRAL CLUSTERING ON DRAWS")
print("="*60)

# Cluster draws by their number patterns
from sklearn.preprocessing import StandardScaler

# Features: the 5 sorted numbers + sum + spread + even_count
sums = numbers.sum(axis=1)
spreads = numbers[:, 4] - numbers[:, 0]
even_counts = np.sum(numbers % 2 == 0, axis=1)
features = np.column_stack([numbers, sums, spreads, even_counts])
features_scaled = StandardScaler().fit_transform(features)

for n_clusters in [3, 5, 8]:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(features_scaled)

    # Does cluster membership predict star2?
    star2_by_cluster = {}
    for c in range(n_clusters):
        mask = labels == c
        s2_in_cluster = star2[mask]
        star2_by_cluster[c] = s2_in_cluster.mean()

    # Temporal structure: are clusters temporally clustered?
    cluster_changes = np.sum(labels[:-1] != labels[1:])
    expected_changes = N * (1 - 1/n_clusters)

    print(f"\nK={n_clusters} clusters:")
    print(f"  Cluster sizes: {[int((labels==c).sum()) for c in range(n_clusters)]}")
    print(f"  Star2 means by cluster: {[f'{v:.2f}' for v in star2_by_cluster.values()]}")
    print(f"  Cluster transitions: {cluster_changes} (expected ~{expected_changes:.0f})")

# ============================================================
# 15. NUMBER EMBEDDINGS VIA CO-OCCURRENCE
# ============================================================
print("\n" + "="*60)
print("15. NUMBER EMBEDDINGS (MDS from co-occurrence)")
print("="*60)

# Build co-occurrence matrix
from itertools import combinations
cooc = np.zeros((50, 50))
for draw in numbers:
    for i, j in combinations(range(5), 2):
        a, b = draw[i] - 1, draw[j] - 1
        cooc[a][b] += 1
        cooc[b][a] += 1

# Convert to distance: distance = 1 / (cooc + 1)
dist_matrix = 1.0 / (cooc + 1)
np.fill_diagonal(dist_matrix, 0)

# MDS to 2D
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
embedding = mds.fit_transform(dist_matrix)

# Cluster the embeddings
km_emb = KMeans(n_clusters=5, n_init=10, random_state=42)
emb_labels = km_emb.fit_predict(embedding)

print("Number clusters from co-occurrence embedding:")
for c in range(5):
    nums_in_cluster = [i+1 for i in range(50) if emb_labels[i] == c]
    print(f"  Cluster {c}: {nums_in_cluster}")

# Check if embedding clusters have temporal structure
print("\nEmbedding cluster temporal analysis:")
for c in range(5):
    cluster_nums = set(i+1 for i in range(50) if emb_labels[i] == c)
    # Count how many numbers from each draw fall in this cluster
    counts_per_draw = np.array([sum(1 for n in draw if n in cluster_nums) for draw in numbers])
    mean_early = counts_per_draw[:N//2].mean()
    mean_late = counts_per_draw[N//2:].mean()
    print(f"  Cluster {c}: early_mean={mean_early:.3f}, late_mean={mean_late:.3f}")

# ============================================================
# 16. SYMBOLIC REGRESSION ATTEMPT ON STAR2
# ============================================================
print("\n" + "="*60)
print("16. SYMBOLIC REGRESSION ON STAR2")
print("="*60)

# Try simple formulas: star2[t] = f(star2[t-1], star2[t-2], ...)
# Test various simple models
print("\n--- Linear models ---")
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error

# Build features: lags 1-10
max_lag = 10
X_lags = np.column_stack([star2[max_lag-l:-l] for l in range(1, max_lag+1)])
y_target = star2[max_lag:]

# Train/test split
split = int(0.8 * len(y_target))
X_train, X_test = X_lags[:split], X_lags[split:]
y_train, y_test = y_target[:split], y_target[split:]

# Linear regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = lr.score(X_test, y_test)
print(f"Linear (10 lags): MAE={mae_lr:.3f}, R²={r2_lr:.4f}")
print(f"  Coefficients: {lr.coef_.round(4)}")

# Baseline: predict mean
mae_base = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
print(f"Baseline (mean): MAE={mae_base:.3f}")

# Round predictions to nearest integer for accuracy
y_pred_rounded = np.clip(np.round(y_pred_lr), 1, 12).astype(int)
acc_lr = (y_pred_rounded == y_test).mean()
print(f"Linear rounded accuracy: {100*acc_lr:.2f}%")

# --- Modular arithmetic patterns ---
print("\n--- Modular arithmetic ---")
# Test: star2[t] mod k has structure?
for k in [2, 3, 4, 6]:
    s2_mod = star2 % k
    # Check if mod-k sequence has autocorrelation
    r = np.corrcoef(s2_mod[:-1], s2_mod[1:])[0, 1]
    # Mode prediction
    correct = 0
    for i in range(100, N):
        recent_mod = s2_mod[i-100:i]
        pred_mod = Counter(recent_mod.tolist()).most_common(1)[0][0]
        if s2_mod[i] == pred_mod:
            correct += 1
    acc = correct / (N - 100)
    baseline = 1/k
    z = (acc - baseline) / np.sqrt(baseline * (1-baseline) / (N-100))
    print(f"  mod {k}: autocorr={r:+.4f}, rolling_mode_acc={100*acc:.1f}% (baseline={100*baseline:.1f}%, z={z:+.1f}σ)")

# --- Difference patterns ---
print("\n--- Difference patterns ---")
diff1 = np.diff(star2.astype(int))
print(f"Star2 diff(1): mean={diff1.mean():.3f}, std={diff1.std():.3f}")
for lag in [1, 2, 3]:
    r = np.corrcoef(diff1[:-lag], diff1[lag:])[0, 1]
    print(f"  diff autocorr lag {lag}: r={r:+.4f}")

# Second difference
diff2 = np.diff(diff1)
print(f"Star2 diff(2): mean={diff2.mean():.3f}, std={diff2.std():.3f}")
for lag in [1, 2, 3]:
    r = np.corrcoef(diff2[:-lag], diff2[lag:])[0, 1]
    print(f"  diff2 autocorr lag {lag}: r={r:+.4f}")

# ============================================================
# 17. JACKPOT SIZE EFFECT
# ============================================================
print("\n" + "="*60)
print("17. JACKPOT SIZE EFFECT")
print("="*60)

jackpots = data['jackpots']
has_winner = data['has_winner']

# Does jackpot size correlate with anything?
nonzero_mask = jackpots > 0
jp_nonzero = jackpots[nonzero_mask]
print(f"Draws with jackpot info: {nonzero_mask.sum()}")
print(f"Jackpot range: {jp_nonzero.min():.0f} - {jp_nonzero.max():.0f}")

if nonzero_mask.sum() > 100:
    # Correlation with stars
    s2_jp = star2[nonzero_mask]
    r_jp = np.corrcoef(jp_nonzero, s2_jp)[0, 1]
    print(f"Correlation jackpot vs star2: r={r_jp:+.4f}")

    # High vs low jackpot draws
    median_jp = np.median(jp_nonzero)
    high_jp = jp_nonzero > median_jp
    s2_high = s2_jp[high_jp].mean()
    s2_low = s2_jp[~high_jp].mean()
    print(f"Star2 mean (high jackpot): {s2_high:.2f}")
    print(f"Star2 mean (low jackpot):  {s2_low:.2f}")

    # Number of winners vs jackpot
    hw_jp = has_winner[nonzero_mask]
    print(f"Winner rate (high jp): {hw_jp[high_jp].mean():.3f}")
    print(f"Winner rate (low jp):  {hw_jp[~high_jp].mean():.3f}")

# ============================================================
# 18. DAY-OF-WEEK EFFECT
# ============================================================
print("\n" + "="*60)
print("18. DAY-OF-WEEK / TEMPORAL EFFECTS")
print("="*60)

from datetime import datetime

days = [datetime.strptime(d, '%Y-%m-%d').weekday() for d in dates]
days = np.array(days)
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

print("Draws per day of week:")
for d in range(7):
    mask = days == d
    n_d = mask.sum()
    if n_d > 0:
        s2_mean = star2[mask].mean()
        s2_mode = Counter(star2[mask].tolist()).most_common(1)[0][0]
        num_mean = numbers[mask].mean()
        print(f"  {day_names[d]}: n={n_d:4d}, star2_mean={s2_mean:.2f}, star2_mode={s2_mode}, nums_mean={num_mean:.2f}")

# ============================================================
# 19. COMBINED PREDICTOR — STAR2
# ============================================================
print("\n" + "="*60)
print("19. COMBINED / ENHANCED STAR2 PREDICTOR")
print("="*60)

# Can we beat rolling mode by combining features?
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Features for predicting star2[t]
print("\n--- Random Forest predictor ---")
feature_names = []
X_all = []

# Lags of star2
for lag in range(1, 11):
    X_all.append(star2[10-lag:-lag] if lag < 10 else star2[:-(10)])
    feature_names.append(f"s2_lag{lag}")

# Rolling stats
for w in [20, 50, 100]:
    rolling_mean = np.array([star2[max(0,i-w):i].mean() for i in range(10, N)])
    rolling_std = np.array([star2[max(0,i-w):i].std() for i in range(10, N)])
    rolling_mode_val = np.array([Counter(star2[max(0,i-w):i].tolist()).most_common(1)[0][0] for i in range(10, N)])
    X_all.extend([rolling_mean, rolling_std, rolling_mode_val])
    feature_names.extend([f"mean_w{w}", f"std_w{w}", f"mode_w{w}"])

# Star1 lags
for lag in range(1, 4):
    X_all.append(star1[10-lag:-lag] if lag < 10 else star1[:-(10)])
    feature_names.append(f"s1_lag{lag}")

# Number features
sums_arr = numbers.sum(axis=1)[10:]
spreads_arr = (numbers[:, 4] - numbers[:, 0])[10:]
X_all.extend([sums_arr, spreads_arr])
feature_names.extend(["num_sum", "num_spread"])

X_combined = np.column_stack(X_all)
y_combined = star2[10:]

# Ensure lengths match
min_len = min(len(X_combined), len(y_combined))
X_combined = X_combined[:min_len]
y_combined = y_combined[:min_len]

# Time-based split
split = int(0.75 * len(y_combined))
X_tr, X_te = X_combined[:split], X_combined[split:]
y_tr, y_te = y_combined[:split], y_combined[split:]

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_te)
acc_rf = accuracy_score(y_te, y_pred_rf)
z_rf = (acc_rf - 1/12) / np.sqrt((1/12)*(11/12)/len(y_te))
print(f"RF accuracy: {100*acc_rf:.2f}% (baseline 8.33%, z={z_rf:+.1f}σ)")

# Feature importances
importances = rf.feature_importances_
top_feats = np.argsort(importances)[::-1][:10]
print(f"Top features: {[(feature_names[i], f'{importances[i]:.3f}') for i in top_feats]}")

# Gradient Boosting
print("\n--- Gradient Boosting predictor ---")
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
gb.fit(X_tr, y_tr)
y_pred_gb = gb.predict(X_te)
acc_gb = accuracy_score(y_te, y_pred_gb)
z_gb = (acc_gb - 1/12) / np.sqrt((1/12)*(11/12)/len(y_te))
print(f"GB accuracy: {100*acc_gb:.2f}% (baseline 8.33%, z={z_gb:+.1f}σ)")

# Compare to simple rolling mode
correct_rm = 0
for i in range(split+10, N):
    recent = star2[i-100:i]
    mode = Counter(recent.tolist()).most_common(1)[0][0]
    if star2[i] == mode:
        correct_rm += 1
acc_rm = correct_rm / (N - split - 10)
z_rm = (acc_rm - 1/12) / np.sqrt((1/12)*(11/12)/(N-split-10))
print(f"\nRolling mode (w=100) on same period: {100*acc_rm:.2f}% (z={z_rm:+.1f}σ)")

# ============================================================
# 20. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("\n" + "="*60)
print("20. BOOTSTRAP CONFIDENCE INTERVALS FOR ROLLING MODE")
print("="*60)

n_boot = 2000
boot_accs = []
test_indices = list(range(100, N))
for _ in range(n_boot):
    sample = np.random.choice(test_indices, size=len(test_indices), replace=True)
    correct = 0
    for i in sample:
        recent = star2[i-100:i]
        mode = Counter(recent.tolist()).most_common(1)[0][0]
        if star2[i] == mode:
            correct += 1
    boot_accs.append(correct / len(sample))

boot_accs = np.array(boot_accs)
ci_low = np.percentile(boot_accs, 2.5)
ci_high = np.percentile(boot_accs, 97.5)
print(f"Rolling mode accuracy: {boot_accs.mean():.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Baseline: {1/12:.4f}")
print(f"CI entirely above baseline: {ci_low > 1/12}")

print("\n" + "="*60)
print("UNCONVENTIONAL EXPLORATION COMPLETE")
print("="*60)
