"""
Walk-Forward Falsification of GradientBoosting Horse Racing Model
=================================================================
"Tout résultat est suspect jusqu'à falsification"

Tests the claimed +67.6% ROI across 4 consecutive walk-forward windows
plus a permutation test baseline.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & prepare data ──────────────────────────────────────────
print("=" * 80)
print("WALK-FORWARD FALSIFICATION — GradientBoosting Horse Racing Model")
print("=" * 80)

df = pd.read_csv('/home/user/Eurexplo/data_pmu/race_results.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
print(f"\nDataset: {len(df):,} runners, {df['race_id'].nunique():,} races")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── 2. Compute 8 features per runner (grouped by race_id) ───────────
print("\nComputing 8 odds-based features...")

# Per-race rankings
df['odds_rank'] = df.groupby('race_id')['morning_odds'].rank(method='min')
df['final_rank'] = df.groupby('race_id')['final_odds'].rank(method='min')

# Derived features
df['odds_ratio'] = df['final_odds'] / df['morning_odds'].replace(0, np.nan)
df['inv_odds'] = 1.0 / df['final_odds'].replace(0, np.nan)
df['n_partants'] = df['field_size']
df['drift'] = df['odds_drift_pct']

# Target: win = finish_position == 1
df['win'] = (df['finish_position'] == 1).astype(int)

FEATURES = ['final_odds', 'morning_odds', 'drift', 'odds_rank',
            'final_rank', 'odds_ratio', 'inv_odds', 'n_partants']

# Drop rows with missing features or target
df = df.dropna(subset=FEATURES + ['win', 'morning_odds', 'final_odds'])
print(f"After cleaning: {len(df):,} runners, {df['race_id'].nunique():,} races")

# ── 3. Define walk-forward windows by actual months in data ─────────
df['year_month'] = df['date'].dt.to_period('M')
months_sorted = sorted(df['year_month'].unique())
print(f"\nMonths in data ({len(months_sorted)} total): {[str(m) for m in months_sorted]}")

# Define 4 windows
windows = [
    {"name": "W1", "train": months_sorted[0:2], "test": months_sorted[2:3],
     "desc": f"Train {months_sorted[0]}-{months_sorted[1]}, Test {months_sorted[2]}"},
    {"name": "W2", "train": months_sorted[0:3], "test": months_sorted[3:4],
     "desc": f"Train {months_sorted[0]}-{months_sorted[2]}, Test {months_sorted[3]}"},
    {"name": "W3", "train": months_sorted[1:4], "test": months_sorted[4:5],
     "desc": f"Train {months_sorted[1]}-{months_sorted[3]}, Test {months_sorted[4]}"},
    {"name": "W4", "train": months_sorted[2:5], "test": months_sorted[5:6],
     "desc": f"Train {months_sorted[2]}-{months_sorted[4]}, Test {months_sorted[5]}"},
]

print("\nWalk-forward windows:")
for w in windows:
    print(f"  {w['name']}: {w['desc']}")

# ── 4. Walk-forward evaluation function ─────────────────────────────
def evaluate_window(df, train_months, test_months, permute=False, seed=42):
    """Train on train_months, test on test_months. Return metrics dict."""
    train = df[df['year_month'].isin(train_months)].copy()
    test = df[df['year_month'].isin(test_months)].copy()

    if len(train) == 0 or len(test) == 0:
        return None

    X_train = train[FEATURES].values
    y_train = train['win'].values.copy()
    X_test = test[FEATURES].values
    y_test = test['win'].values

    if permute:
        rng = np.random.RandomState(seed)
        y_train = rng.permutation(y_train)

    # Train GradientBoosting
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
        learning_rate=0.1
    )
    model.fit(X_train, y_train)

    # Predict probabilities on test
    probs = model.predict_proba(X_test)[:, 1]
    test = test.copy()
    test['pred_prob'] = probs

    # AUC
    try:
        auc = roc_auc_score(y_test, probs)
    except ValueError:
        auc = np.nan

    # ── Simulated betting ────────────────────────────────────────
    # For each race: pick highest-prob runner.
    # Bet if edge > 8% (pred_prob - 1/morning_odds > 0.08) AND morning_odds in [3,10]
    total_bets = 0
    total_wins = 0
    total_pnl = 0.0  # net P&L in units staked

    for race_id, group in test.groupby('race_id'):
        if len(group) < 2:
            continue

        # Pick highest predicted probability runner
        best_idx = group['pred_prob'].idxmax()
        best = group.loc[best_idx]

        model_prob = best['pred_prob']
        morning_odds = best['morning_odds']
        final_odds = best['final_odds']
        implied_prob = 1.0 / morning_odds if morning_odds > 0 else 1.0

        edge = model_prob - implied_prob

        # Filter: edge > 8% AND morning_odds in [3, 10]
        if edge > 0.08 and 3.0 <= morning_odds <= 10.0:
            total_bets += 1
            if best['win'] == 1:
                total_wins += 1
                total_pnl += (final_odds - 1)  # net profit on 1-unit stake
            else:
                total_pnl -= 1  # lose the stake

    win_rate = total_wins / total_bets if total_bets > 0 else 0.0
    roi = total_pnl / total_bets if total_bets > 0 else 0.0

    return {
        'train_size': len(train),
        'test_size': len(test),
        'test_races': test['race_id'].nunique(),
        'auc': auc,
        'bets': total_bets,
        'wins': total_wins,
        'win_rate': win_rate,
        'pnl': total_pnl,
        'roi': roi,
    }

# ── 5. Run all windows ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("RUNNING WALK-FORWARD VALIDATION")
print("=" * 80)

results = []
for w in windows:
    print(f"\n--- {w['name']}: {w['desc']} ---")
    res = evaluate_window(df, w['train'], w['test'], permute=False)
    if res:
        res['window'] = w['name']
        res['desc'] = w['desc']
        res['type'] = 'MODEL'
        results.append(res)
        print(f"  Train: {res['train_size']:,} runners | Test: {res['test_size']:,} runners ({res['test_races']} races)")
        print(f"  AUC: {res['auc']:.4f}")
        print(f"  Bets: {res['bets']} | Wins: {res['wins']} | Win rate: {res['win_rate']:.1%}")
        print(f"  PnL: {res['pnl']:+.2f} units | ROI: {res['roi']:+.1%}")

# ── 6. Permutation test (randomized labels) ─────────────────────────
print("\n" + "=" * 80)
print("PERMUTATION TEST (RANDOMIZED LABELS)")
print("=" * 80)

perm_results = []
N_PERMS = 5  # multiple seeds for robustness
for w in windows:
    perm_rois = []
    perm_aucs = []
    for seed in range(N_PERMS):
        pres = evaluate_window(df, w['train'], w['test'], permute=True, seed=seed)
        if pres:
            perm_rois.append(pres['roi'])
            perm_aucs.append(pres['auc'])

    avg_roi = np.mean(perm_rois) if perm_rois else np.nan
    avg_auc = np.mean(perm_aucs) if perm_aucs else np.nan
    avg_bets = np.mean([evaluate_window(df, w['train'], w['test'], permute=True, seed=s)['bets']
                        for s in range(N_PERMS)])

    perm_results.append({
        'window': w['name'],
        'desc': w['desc'],
        'type': 'PERMUTED',
        'avg_auc': avg_auc,
        'avg_roi': avg_roi,
        'avg_bets': avg_bets,
    })
    print(f"\n--- {w['name']} (permuted, avg of {N_PERMS} seeds) ---")
    print(f"  Avg AUC: {avg_auc:.4f} | Avg ROI: {avg_roi:+.1%} | Avg bets: {avg_bets:.1f}")

# ── 7. Summary table ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY TABLE — WALK-FORWARD FALSIFICATION")
print("=" * 80)

header = f"{'Window':<6} {'Period':<40} {'Type':<8} {'AUC':>6} {'Bets':>5} {'Wins':>5} {'WinR%':>6} {'PnL':>8} {'ROI%':>8}"
print(header)
print("-" * len(header))

for r in results:
    print(f"{r['window']:<6} {r['desc']:<40} {'MODEL':<8} {r['auc']:>6.4f} {r['bets']:>5} {r['wins']:>5} {r['win_rate']:>5.1%} {r['pnl']:>+8.2f} {r['roi']:>+7.1%}")

print("-" * len(header))
for p in perm_results:
    print(f"{p['window']:<6} {p['desc']:<40} {'PERMUT':<8} {p['avg_auc']:>6.4f} {p['avg_bets']:>5.0f} {'—':>5} {'—':>6} {'—':>8} {p['avg_roi']:>+7.1%}")

# ── 8. Aggregate analysis ───────────────────────────────────────────
print("\n" + "=" * 80)
print("AGGREGATE ANALYSIS")
print("=" * 80)

total_bets = sum(r['bets'] for r in results)
total_wins = sum(r['wins'] for r in results)
total_pnl = sum(r['pnl'] for r in results)
overall_roi = total_pnl / total_bets if total_bets > 0 else 0
overall_wr = total_wins / total_bets if total_bets > 0 else 0
aucs = [r['auc'] for r in results if not np.isnan(r['auc'])]
rois = [r['roi'] for r in results]

print(f"\nAcross all 4 windows:")
print(f"  Total bets:     {total_bets}")
print(f"  Total wins:     {total_wins}")
print(f"  Overall WinR:   {overall_wr:.1%}")
print(f"  Total PnL:      {total_pnl:+.2f} units")
print(f"  Overall ROI:    {overall_roi:+.1%}")
print(f"  Mean AUC:       {np.mean(aucs):.4f} (std: {np.std(aucs):.4f})")
print(f"  ROI range:      {min(rois):+.1%} to {max(rois):+.1%}")
print(f"  ROI std:        {np.std(rois):.1%}")

# Count how many windows are profitable
profitable = sum(1 for r in results if r['roi'] > 0)
print(f"\n  Profitable windows: {profitable}/4")

# Verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
if profitable == 4 and overall_roi > 0.2:
    print("  The +67.6% ROI SURVIVES walk-forward validation across all windows.")
    print("  The signal appears robust (but sample size and market impact remain concerns).")
elif profitable >= 3 and overall_roi > 0.1:
    print("  PARTIAL SURVIVAL: Most windows are profitable but results are uneven.")
    print("  The +67.6% ROI may be inflated by one strong period.")
elif profitable >= 2:
    print("  WEAK SIGNAL: Only some windows profitable. The ROI is likely overstated.")
    print("  High variance suggests the +67.6% ROI is fragile / period-dependent.")
else:
    print("  FALSIFIED: The +67.6% ROI does NOT survive walk-forward validation.")
    print("  The original result was likely an artifact of favorable test-period selection.")

# Check permutation test
perm_avg_roi = np.mean([p['avg_roi'] for p in perm_results])
model_avg_roi = np.mean(rois)
print(f"\n  Model avg ROI: {model_avg_roi:+.1%} vs Permuted avg ROI: {perm_avg_roi:+.1%}")
if model_avg_roi > perm_avg_roi + 0.05:
    print("  Signal significantly exceeds random baseline (>5pp gap).")
else:
    print("  WARNING: Signal does NOT clearly exceed random baseline.")

print("\n" + "=" * 80)
