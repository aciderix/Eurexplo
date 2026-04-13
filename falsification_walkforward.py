"""
Falsification of +67.6% ROI claim via walk-forward validation.
Principle: "tout résultat est suspect jusqu'à falsification"

We test the GradientBoosting model across 4 consecutive time windows
plus a permutation baseline to detect spurious signal.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ─── 1. Load data ────────────────────────────────────────────────────────────
print("=" * 80)
print("FALSIFICATION: Walk-Forward Validation of GradientBoosting ROI Claim")
print("=" * 80)

df = pd.read_csv('/home/user/Eurexplo/data_pmu/race_results.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
print(f"\nTotal runners loaded: {len(df):,}")
print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")

# ─── 2. Feature engineering (8 odds-based features) ─────────────────────────
print("\nComputing 8 odds-based features per runner (grouped by race_id)...")

# Basic features available per runner
df['drift'] = df['odds_drift_pct']
df['odds_ratio'] = df['final_odds'] / df['morning_odds']
df['inv_odds'] = 1.0 / df['final_odds']
df['n_partants'] = df['field_size']

# Rank features (within each race)
df['odds_rank'] = df.groupby('race_id')['morning_odds'].rank(method='min')
df['final_rank'] = df.groupby('race_id')['final_odds'].rank(method='min')

# Target: did this runner win?
df['won'] = (df['finish_position'] == 1).astype(int)

FEATURES = ['final_odds', 'morning_odds', 'drift', 'odds_rank',
            'final_rank', 'odds_ratio', 'inv_odds', 'n_partants']

# Drop rows with missing features
df = df.dropna(subset=FEATURES + ['won', 'morning_odds', 'final_odds'])
print(f"Runners after dropna: {len(df):,}")

# ─── 3. Define time windows ─────────────────────────────────────────────────
# Find actual month boundaries in the data
df['year_month'] = df['date'].dt.to_period('M')
months_sorted = sorted(df['year_month'].unique())
print(f"\nMonths in dataset ({len(months_sorted)} total):")
for m in months_sorted:
    count = (df['year_month'] == m).sum()
    print(f"  {m}: {count:,} runners")

# We need at least 6 months for 4 windows
if len(months_sorted) < 6:
    print(f"\nWARNING: Only {len(months_sorted)} months available. Adjusting windows.")

# Define windows based on actual months
windows = []
if len(months_sorted) >= 6:
    windows = [
        {"name": "W1", "train": months_sorted[0:2], "test": [months_sorted[2]]},
        {"name": "W2", "train": months_sorted[0:3], "test": [months_sorted[3]]},
        {"name": "W3", "train": months_sorted[1:4], "test": [months_sorted[4]]},
        {"name": "W4", "train": months_sorted[2:5], "test": [months_sorted[5]]},
    ]
elif len(months_sorted) >= 4:
    # Fallback: smaller windows
    windows = [
        {"name": "W1", "train": months_sorted[0:2], "test": [months_sorted[2]]},
        {"name": "W2", "train": months_sorted[0:3], "test": [months_sorted[3]]},
    ]
    if len(months_sorted) >= 5:
        windows.append(
            {"name": "W3", "train": months_sorted[1:4], "test": [months_sorted[4]]}
        )
else:
    # Very few months: split into halves
    mid = len(months_sorted) // 2
    windows = [
        {"name": "W1", "train": months_sorted[:mid], "test": months_sorted[mid:]},
    ]

print(f"\n{'='*80}")
print(f"Walk-forward windows defined: {len(windows)}")
for w in windows:
    train_str = f"{w['train'][0]}..{w['train'][-1]}"
    test_str = f"{w['test'][0]}..{w['test'][-1]}"
    print(f"  {w['name']}: Train [{train_str}] → Test [{test_str}]")
print(f"{'='*80}")


# ─── 4. Simulation function ─────────────────────────────────────────────────
def run_window(df, train_months, test_months, features, permute=False, seed=42):
    """Train on train_months, test on test_months. Return metrics dict."""

    train_mask = df['year_month'].isin(train_months)
    test_mask = df['year_month'].isin(test_months)

    X_train = df.loc[train_mask, features].values
    y_train = df.loc[train_mask, 'won'].values
    X_test = df.loc[test_mask, features].values
    y_test = df.loc[test_mask, 'won'].values

    if permute:
        rng = np.random.RandomState(seed)
        y_train = rng.permutation(y_train)

    # Train model
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
        verbose=0
    )
    clf.fit(X_train, y_train)

    # Predict probabilities on test
    proba = clf.predict_proba(X_test)[:, 1]

    # AUC
    try:
        auc = roc_auc_score(y_test, proba)
    except:
        auc = np.nan

    # ── Betting simulation ──
    test_df = df.loc[test_mask].copy()
    test_df['model_prob'] = proba

    total_bets = 0
    total_wins = 0
    total_pnl = 0.0

    for race_id, group in test_df.groupby('race_id'):
        # Pick highest-prob runner in the race
        best_idx = group['model_prob'].idxmax()
        best = group.loc[best_idx]

        model_prob = best['model_prob']
        morning_odds = best['morning_odds']
        final_odds = best['final_odds']
        implied_prob = 1.0 / morning_odds if morning_odds > 0 else 1.0
        edge = model_prob - implied_prob

        # Betting filter: edge > 8% AND morning_odds in [3, 10]
        if edge > 0.08 and 3.0 <= morning_odds <= 10.0:
            total_bets += 1
            if best['won'] == 1:
                total_wins += 1
                total_pnl += (final_odds - 1.0)  # net profit (1 unit staked)
            else:
                total_pnl -= 1.0  # lost the stake

    win_rate = total_wins / total_bets if total_bets > 0 else 0
    roi = total_pnl / total_bets if total_bets > 0 else 0

    return {
        'auc': auc,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_races_test': test_df['race_id'].nunique(),
        'n_bets': total_bets,
        'n_wins': total_wins,
        'win_rate': win_rate,
        'pnl': total_pnl,
        'roi': roi,
    }


# ─── 5. Run all windows ─────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("Running walk-forward validation (real labels)...")
print("─" * 80)

results = []
for w in windows:
    r = run_window(df, w['train'], w['test'], FEATURES, permute=False)
    r['window'] = w['name']
    r['type'] = 'REAL'
    train_str = f"{w['train'][0]}..{w['train'][-1]}"
    test_str = f"{w['test'][0]}..{w['test'][-1]}"
    r['train_period'] = train_str
    r['test_period'] = test_str
    results.append(r)
    print(f"  {w['name']} done: AUC={r['auc']:.4f}, Bets={r['n_bets']}, "
          f"Wins={r['n_wins']}, WinRate={r['win_rate']:.1%}, "
          f"ROI={r['roi']:.1%}, PnL={r['pnl']:.1f}")

# ─── 6. Permutation test (randomized labels) ────────────────────────────────
print("\n" + "─" * 80)
print("Running permutation test (randomized labels)...")
print("─" * 80)

N_PERMUTATIONS = 5
for w in windows:
    perm_rois = []
    perm_aucs = []
    for seed in range(N_PERMUTATIONS):
        r = run_window(df, w['train'], w['test'], FEATURES, permute=True, seed=seed)
        perm_rois.append(r['roi'])
        perm_aucs.append(r['auc'])

    avg_roi = np.mean(perm_rois)
    avg_auc = np.mean(perm_aucs)
    r_perm = {
        'window': w['name'],
        'type': 'PERMUTED',
        'auc': avg_auc,
        'n_bets': '-',
        'n_wins': '-',
        'win_rate': '-',
        'pnl': '-',
        'roi': avg_roi,
        'train_period': f"{w['train'][0]}..{w['train'][-1]}",
        'test_period': f"{w['test'][0]}..{w['test'][-1]}",
        'n_train': '-',
        'n_test': '-',
        'n_races_test': '-',
    }
    results.append(r_perm)
    print(f"  {w['name']} permuted: avg AUC={avg_auc:.4f}, avg ROI={avg_roi:.1%}")


# ─── 7. Summary table ───────────────────────────────────────────────────────
print("\n")
print("=" * 100)
print("FALSIFICATION RESULTS — Walk-Forward Validation Summary")
print("=" * 100)
print(f"{'Window':<8} {'Type':<10} {'Train Period':<22} {'Test Period':<14} "
      f"{'AUC':<8} {'Bets':<6} {'Wins':<6} {'WinRate':<9} {'PnL':<10} {'ROI':<10}")
print("-" * 100)

real_results = [r for r in results if r['type'] == 'REAL']
perm_results = [r for r in results if r['type'] == 'PERMUTED']

for r in real_results:
    wr = f"{r['win_rate']:.1%}" if isinstance(r['win_rate'], float) else r['win_rate']
    pnl = f"{r['pnl']:.2f}" if isinstance(r['pnl'], float) else r['pnl']
    roi = f"{r['roi']:.1%}" if isinstance(r['roi'], float) else r['roi']
    auc = f"{r['auc']:.4f}" if isinstance(r['auc'], float) else r['auc']
    print(f"{r['window']:<8} {r['type']:<10} {r['train_period']:<22} {r['test_period']:<14} "
          f"{auc:<8} {str(r['n_bets']):<6} {str(r['n_wins']):<6} {wr:<9} {pnl:<10} {roi:<10}")

print("-" * 100)
for r in perm_results:
    roi = f"{r['roi']:.1%}" if isinstance(r['roi'], float) else r['roi']
    auc = f"{r['auc']:.4f}" if isinstance(r['auc'], float) else r['auc']
    print(f"{r['window']:<8} {r['type']:<10} {r['train_period']:<22} {r['test_period']:<14} "
          f"{auc:<8} {'(avg)':<6} {'(avg)':<6} {'(avg)':<9} {'(avg)':<10} {roi:<10}")

print("=" * 100)

# ─── 8. Verdict ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

roi_values = [r['roi'] for r in real_results if isinstance(r['roi'], float)]
auc_values = [r['auc'] for r in real_results if isinstance(r['auc'], float)]
bet_counts = [r['n_bets'] for r in real_results if isinstance(r['n_bets'], int)]

if roi_values:
    avg_roi = np.mean(roi_values)
    min_roi = np.min(roi_values)
    max_roi = np.max(roi_values)
    std_roi = np.std(roi_values)
    all_positive = all(r > 0 for r in roi_values)

    print(f"\nROI across windows:  min={min_roi:.1%}  max={max_roi:.1%}  "
          f"mean={avg_roi:.1%}  std={std_roi:.1%}")
    print(f"AUC across windows:  min={np.min(auc_values):.4f}  max={np.max(auc_values):.4f}  "
          f"mean={np.mean(auc_values):.4f}")
    print(f"Bets across windows: min={np.min(bet_counts)}  max={np.max(bet_counts)}  "
          f"total={np.sum(bet_counts)}")

    # Permutation AUCs for comparison
    perm_aucs_all = [r['auc'] for r in perm_results if isinstance(r['auc'], float)]
    if perm_aucs_all:
        print(f"\nPermutation baseline AUC: {np.mean(perm_aucs_all):.4f} "
              f"(real model should be significantly above this)")

    perm_rois_all = [r['roi'] for r in perm_results if isinstance(r['roi'], float)]
    if perm_rois_all:
        print(f"Permutation baseline ROI: {np.mean(perm_rois_all):.1%}")

    print(f"\n--- Key question: does +67.6% ROI survive all windows? ---")

    if all_positive and min_roi > 0.20:
        print(f"RESULT: ROI is POSITIVE in ALL windows (min={min_roi:.1%}).")
        print(f"The signal appears ROBUST — but {len(windows)} windows is still limited.")
        print(f"The +67.6% ROI claim SURVIVES initial falsification.")
    elif all_positive:
        print(f"RESULT: ROI is positive in all windows but drops as low as {min_roi:.1%}.")
        print(f"The +67.6% figure likely overstates the true edge.")
        print(f"A more conservative estimate would be ~{avg_roi:.1%} ROI.")
    elif any(r > 0 for r in roi_values):
        n_pos = sum(1 for r in roi_values if r > 0)
        print(f"RESULT: ROI is positive in only {n_pos}/{len(roi_values)} windows.")
        print(f"The +67.6% ROI claim is PARTIALLY FALSIFIED — not stable across time.")
    else:
        print(f"RESULT: ROI is NEGATIVE in ALL windows.")
        print(f"The +67.6% ROI claim is FALSIFIED — likely an artifact of overfitting.")

    # Variance check
    if std_roi > 0.30:
        print(f"\nWARNING: High ROI variance (std={std_roi:.1%}) suggests the strategy is")
        print(f"highly sensitive to the test period. This is a red flag for robustness.")

    # Sample size check
    if bet_counts and np.min(bet_counts) < 30:
        print(f"\nWARNING: Some windows have very few bets (min={np.min(bet_counts)}). "
              f"Small sample size makes ROI estimates unreliable.")

print(f"\n{'='*80}")
print("Falsification protocol complete.")
print(f"{'='*80}")
