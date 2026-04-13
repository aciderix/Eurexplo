"""
PMU Horse Racing Feature Analysis
==================================
Explores 68 pre-computed features, trains GradientBoosting with temporal split,
and answers: do jockey/trainer/horse features add value beyond odds alone?
"""

import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 80)
print("1. LOADING DATASET")
print("=" * 80)

df = pd.read_csv('/home/user/Eurexplo/data_pmu/export-1-pt-utf.csv')
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")

# Parse date
df['cr-Date'] = pd.to_datetime(df['cr-Date'], format='%d-%m-%Y', dayfirst=True)
print(f"\nDate range: {df['cr-Date'].min()} to {df['cr-Date'].max()}")

# ============================================================
# 2. CREATE TARGET
# ============================================================
print("\n" + "=" * 80)
print("2. TARGET VARIABLE")
print("=" * 80)

df['winner'] = (df['place'] == 1).astype(int)
print(f"Winners: {df['winner'].sum()} ({df['winner'].mean()*100:.2f}%)")
print(f"Non-winners: {(df['winner'] == 0).sum()} ({(1-df['winner'].mean())*100:.2f}%)")

# ============================================================
# 3. IDENTIFY NUMERIC FEATURES
# ============================================================
# Exclude target-related, identifiers, and categorical columns
exclude_cols = ['place', 'placeoptin', 'rapport', 'winner', 'gains',
                'cr-reunion', 'cr-num', 'cr-Hippodrome', 'cr-Evt', 'cr-Date',
                'cr-autostart', 'cr-corde', 'cr-etat du terrain', 'cr-distance']

numeric_features = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
print(f"\nNumeric features for analysis: {len(numeric_features)}")
print(f"Features: {numeric_features}")

# ============================================================
# 3. FEATURE CORRELATIONS & DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 80)
print("3. FEATURE EXPLORATION: CORRELATION WITH WINNING")
print("=" * 80)

# Point-biserial correlation (target is binary)
corr_results = []
for feat in numeric_features:
    valid = df[[feat, 'winner']].dropna()
    if len(valid) > 10 and valid[feat].std() > 0:
        r, p = pointbiserialr(valid['winner'], valid[feat])
        corr_results.append({'feature': feat, 'correlation': r, 'p_value': p, 'abs_corr': abs(r)})

corr_df = pd.DataFrame(corr_results).sort_values('abs_corr', ascending=False)
print("\nPoint-biserial correlations with winner (sorted by |r|):")
print("-" * 65)
print(f"{'Feature':<45} {'r':>8} {'p-value':>10}")
print("-" * 65)
for _, row in corr_df.iterrows():
    sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
    print(f"{row['feature']:<45} {row['correlation']:>8.4f} {row['p_value']:>10.2e} {sig}")

# Winner vs Loser distributions for top features
print("\n" + "-" * 80)
print("FEATURE DISTRIBUTIONS: Winners vs Losers (Top 20 by |correlation|)")
print("-" * 80)
top20 = corr_df.head(20)['feature'].tolist()
for feat in top20:
    w = df.loc[df['winner'] == 1, feat].dropna()
    l = df.loc[df['winner'] == 0, feat].dropna()
    print(f"\n  {feat}:")
    print(f"    Winners  -> mean={w.mean():.4f}, median={w.median():.4f}, std={w.std():.4f}")
    print(f"    Losers   -> mean={l.mean():.4f}, median={l.median():.4f}, std={l.std():.4f}")
    print(f"    Diff     -> {w.mean() - l.mean():+.4f} (winners {'higher' if w.mean() > l.mean() else 'lower'})")

# ============================================================
# Mutual Information
# ============================================================
print("\n" + "=" * 80)
print("MUTUAL INFORMATION SCORES")
print("=" * 80)

X_mi = df[numeric_features].fillna(0)
y_mi = df['winner']
mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42, n_neighbors=5)
mi_df = pd.DataFrame({'feature': numeric_features, 'MI': mi_scores}).sort_values('MI', ascending=False)

print(f"\n{'Feature':<45} {'MI Score':>10}")
print("-" * 57)
for _, row in mi_df.iterrows():
    bar = "#" * int(row['MI'] * 200)
    print(f"{row['feature']:<45} {row['MI']:>10.4f}  {bar}")

# ============================================================
# TOP 15 MOST PREDICTIVE FEATURES (combined ranking)
# ============================================================
print("\n" + "=" * 80)
print("TOP 15 MOST PREDICTIVE FEATURES (combined MI + |correlation| ranking)")
print("=" * 80)

# Rank by MI and by |corr|, then average ranks
mi_rank = mi_df.reset_index(drop=True).reset_index().rename(columns={'index': 'mi_rank'})
mi_rank['mi_rank'] = mi_rank['mi_rank'] + 1
corr_rank = corr_df.reset_index(drop=True).reset_index().rename(columns={'index': 'corr_rank'})
corr_rank['corr_rank'] = corr_rank['corr_rank'] + 1

combined = mi_rank[['feature', 'MI', 'mi_rank']].merge(
    corr_rank[['feature', 'correlation', 'abs_corr', 'corr_rank']], on='feature')
combined['avg_rank'] = (combined['mi_rank'] + combined['corr_rank']) / 2
combined = combined.sort_values('avg_rank')

print(f"\n{'Rank':<5} {'Feature':<45} {'MI':>7} {'|r|':>7} {'AvgRank':>8}")
print("-" * 75)
for i, (_, row) in enumerate(combined.head(15).iterrows(), 1):
    print(f"{i:<5} {row['feature']:<45} {row['MI']:>7.4f} {row['abs_corr']:>7.4f} {row['avg_rank']:>8.1f}")

top15_features = combined.head(15)['feature'].tolist()

# ============================================================
# 4. TEMPORAL TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 80)
print("4. TEMPORAL TRAIN/TEST SPLIT")
print("=" * 80)

df_sorted = df.sort_values('cr-Date').reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.7)
split_date = df_sorted.loc[split_idx, 'cr-Date']

train = df_sorted.iloc[:split_idx].copy()
test = df_sorted.iloc[split_idx:].copy()

print(f"Train: {len(train)} rows, dates {train['cr-Date'].min()} to {train['cr-Date'].max()}")
print(f"Test:  {len(test)} rows, dates {test['cr-Date'].min()} to {test['cr-Date'].max()}")
print(f"Split date: {split_date}")
print(f"Train winner rate: {train['winner'].mean()*100:.2f}%")
print(f"Test  winner rate: {test['winner'].mean()*100:.2f}%")

# ============================================================
# 5. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 80)
print("5. GRADIENT BOOSTING MODELS")
print("=" * 80)

# --- Model A: ALL numeric features ---
all_feats = numeric_features
X_train_all = train[all_feats].fillna(0)
X_test_all = test[all_feats].fillna(0)
y_train = train['winner']
y_test = test['winner']

gb_all = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, min_samples_leaf=20
)
gb_all.fit(X_train_all, y_train)
proba_all = gb_all.predict_proba(X_test_all)[:, 1]
auc_all = roc_auc_score(y_test, proba_all)

print(f"\nModel A (ALL {len(all_feats)} features): AUC = {auc_all:.4f}")

# --- Model B: Top 15 features ---
X_train_t15 = train[top15_features].fillna(0)
X_test_t15 = test[top15_features].fillna(0)

gb_top15 = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, min_samples_leaf=20
)
gb_top15.fit(X_train_t15, y_train)
proba_top15 = gb_top15.predict_proba(X_test_t15)[:, 1]
auc_top15 = roc_auc_score(y_test, proba_top15)

print(f"Model B (Top 15 features):       AUC = {auc_top15:.4f}")

# --- Model C: CoteProbable ONLY (odds-only baseline) ---
X_train_odds = train[['CoteProbable']].fillna(train['CoteProbable'].median())
X_test_odds = test[['CoteProbable']].fillna(test['CoteProbable'].median())

gb_odds = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, min_samples_leaf=20
)
gb_odds.fit(X_train_odds, y_train)
proba_odds = gb_odds.predict_proba(X_test_odds)[:, 1]
auc_odds = roc_auc_score(y_test, proba_odds)

print(f"Model C (CoteProbable only):      AUC = {auc_odds:.4f}")

# --- Model D: All features EXCEPT CoteProbable/coteprobable ---
no_odds_feats = [f for f in all_feats if f.lower() not in ['coteprobable', 'coteprobable']]
X_train_no = train[no_odds_feats].fillna(0)
X_test_no = test[no_odds_feats].fillna(0)

gb_no_odds = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, min_samples_leaf=20
)
gb_no_odds.fit(X_train_no, y_train)
proba_no_odds = gb_no_odds.predict_proba(X_test_no)[:, 1]
auc_no_odds = roc_auc_score(y_test, proba_no_odds)

print(f"Model D (All EXCEPT odds):        AUC = {auc_no_odds:.4f}")

# --- Model E: Jockey/Trainer/Horse features only ---
jtc_feats = [f for f in all_feats if any(k in f.lower() for k in ['jockey', 'entraineur', 'cheval'])]
print(f"\nJockey/Trainer/Horse features ({len(jtc_feats)}): {jtc_feats}")

X_train_jtc = train[jtc_feats].fillna(0)
X_test_jtc = test[jtc_feats].fillna(0)

gb_jtc = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, min_samples_leaf=20
)
gb_jtc.fit(X_train_jtc, y_train)
proba_jtc = gb_jtc.predict_proba(X_test_jtc)[:, 1]
auc_jtc = roc_auc_score(y_test, proba_jtc)

print(f"Model E (Jockey/Trainer/Horse only): AUC = {auc_jtc:.4f}")

# --- Model F: Odds + Jockey/Trainer/Horse features ---
odds_jtc_feats = ['CoteProbable', 'coteprobable'] + jtc_feats
odds_jtc_feats = [f for f in odds_jtc_feats if f in all_feats]
# deduplicate
odds_jtc_feats = list(dict.fromkeys(odds_jtc_feats))

X_train_oj = train[odds_jtc_feats].fillna(0)
X_test_oj = test[odds_jtc_feats].fillna(0)

gb_oj = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42, min_samples_leaf=20
)
gb_oj.fit(X_train_oj, y_train)
proba_oj = gb_oj.predict_proba(X_test_oj)[:, 1]
auc_oj = roc_auc_score(y_test, proba_oj)

print(f"Model F (Odds + J/T/H features):    AUC = {auc_oj:.4f}")

# ============================================================
# Feature importance from Model A (all features)
# ============================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (GradientBoosting, All Features Model)")
print("=" * 80)
importances = pd.Series(gb_all.feature_importances_, index=all_feats).sort_values(ascending=False)
print(f"\n{'Feature':<45} {'Importance':>12}")
print("-" * 59)
for feat, imp in importances.items():
    bar = "#" * int(imp * 300)
    print(f"{feat:<45} {imp:>12.4f}  {bar}")

# ============================================================
# 6. SIMULATED ROI
# ============================================================
print("\n" + "=" * 80)
print("6. SIMULATED ROI (Flat-bet strategy on test set)")
print("=" * 80)

def compute_roi(test_df, probas, strategy_name, top_n_per_race=1, threshold=None):
    """
    Simulate flat betting: for each race, bet on the horse(s) with highest
    predicted probability. Track ROI.
    """
    t = test_df.copy()
    t['proba'] = probas

    # Group by race (cr-Date + cr-reunion + cr-num)
    t['race_id'] = t['cr-Date'].astype(str) + '_' + t['cr-reunion'].astype(str) + '_' + t['cr-num'].astype(str)

    total_bet = 0
    total_return = 0
    wins = 0
    bets = 0

    for race_id, race in t.groupby('race_id'):
        race_sorted = race.sort_values('proba', ascending=False)
        picks = race_sorted.head(top_n_per_race)

        for _, pick in picks.iterrows():
            if threshold is not None and pick['proba'] < threshold:
                continue
            total_bet += 1
            bets += 1
            if pick['winner'] == 1:
                # CoteProbable is the odds; payout = odds * stake
                payout = pick['CoteProbable']
                total_return += payout
                wins += 1

    roi = (total_return - total_bet) / total_bet * 100 if total_bet > 0 else 0
    win_rate = wins / bets * 100 if bets > 0 else 0
    print(f"\n  {strategy_name}:")
    print(f"    Bets: {bets}, Wins: {wins}, Win rate: {win_rate:.1f}%")
    print(f"    Total bet: {total_bet}, Total return: {total_return:.1f}")
    print(f"    ROI: {roi:+.2f}%")
    return roi, win_rate, bets

# Baseline: always pick lowest odds (favorite)
test_copy = test.copy()
test_copy['race_id'] = test_copy['cr-Date'].astype(str) + '_' + test_copy['cr-reunion'].astype(str) + '_' + test_copy['cr-num'].astype(str)

# For favorite baseline, use inverse of odds as "probability"
inv_odds = 1.0 / test_copy['CoteProbable'].replace(0, np.nan).fillna(100)
compute_roi(test, inv_odds.values, "Baseline: Always bet favorite (lowest odds)", top_n_per_race=1)

# GradientBoosting models
compute_roi(test, proba_all, "Model A: All features (top 1 pick per race)", top_n_per_race=1)
compute_roi(test, proba_top15, "Model B: Top 15 features (top 1 pick per race)", top_n_per_race=1)
compute_roi(test, proba_odds, "Model C: Odds only (top 1 pick per race)", top_n_per_race=1)
compute_roi(test, proba_no_odds, "Model D: All except odds (top 1 pick per race)", top_n_per_race=1)
compute_roi(test, proba_jtc, "Model E: Jockey/Trainer/Horse only (top 1 pick per race)", top_n_per_race=1)
compute_roi(test, proba_oj, "Model F: Odds + J/T/H (top 1 pick per race)", top_n_per_race=1)

# Also try selective betting (high confidence only)
print("\n" + "-" * 60)
print("SELECTIVE BETTING (only bet when model confidence > threshold)")
print("-" * 60)
for thresh in [0.15, 0.20, 0.25, 0.30, 0.35]:
    compute_roi(test, proba_all, f"Model A, threshold={thresh}", top_n_per_race=1, threshold=thresh)

# ============================================================
# 7. VALUE OF PRE-COMPUTED FEATURES BEYOND ODDS
# ============================================================
print("\n" + "=" * 80)
print("7. DO JOCKEY/TRAINER/HORSE FEATURES ADD VALUE BEYOND ODDS?")
print("=" * 80)

print("\n  AUC COMPARISON:")
print(f"    Model C (Odds only):              {auc_odds:.4f}")
print(f"    Model E (J/T/H only):             {auc_jtc:.4f}")
print(f"    Model F (Odds + J/T/H):           {auc_oj:.4f}")
print(f"    Model D (All except odds):        {auc_no_odds:.4f}")
print(f"    Model A (All features):           {auc_all:.4f}")
print(f"    Model B (Top 15):                 {auc_top15:.4f}")

auc_lift_jtc = auc_oj - auc_odds
auc_lift_all = auc_all - auc_odds
print(f"\n  AUC LIFT from adding J/T/H features to odds:   {auc_lift_jtc:+.4f}")
print(f"  AUC LIFT from adding ALL features to odds:     {auc_lift_all:+.4f}")

# Check which J/T/H features appear in top importance
print("\n  J/T/H FEATURES IN TOP IMPORTANCE (All-features model):")
jtc_importances = importances[[f for f in importances.index if any(k in f.lower() for k in ['jockey', 'entraineur', 'cheval'])]]
for feat, imp in jtc_importances.items():
    rank = list(importances.index).index(feat) + 1
    print(f"    Rank {rank:>2}: {feat:<45} importance={imp:.4f}")

# Correlation of JTH features with CoteProbable (are they redundant with odds?)
print("\n  CORRELATION OF J/T/H FEATURES WITH CoteProbable:")
for feat in jtc_feats:
    valid = df[['CoteProbable', feat]].dropna()
    if len(valid) > 10 and valid[feat].std() > 0:
        r, p = pointbiserialr(valid['CoteProbable'], valid[feat])
        print(f"    {feat:<45} r={r:+.4f} (p={p:.2e})")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Dataset: {len(df)} rows, {len(numeric_features)} numeric features
Winner base rate: {df['winner'].mean()*100:.1f}%

MODEL PERFORMANCE (AUC on temporal test set):
  Odds only:              {auc_odds:.4f}
  J/T/H features only:   {auc_jtc:.4f}
  Odds + J/T/H:          {auc_oj:.4f}
  All features:           {auc_all:.4f}
  Top 15 features:        {auc_top15:.4f}

KEY FINDING:
  Adding J/T/H features to odds gives AUC lift of {auc_lift_jtc:+.4f}
  Adding ALL features to odds gives AUC lift of   {auc_lift_all:+.4f}

  {'YES' if auc_lift_jtc > 0.005 else 'MARGINAL/NO'}: J/T/H features {'DO' if auc_lift_jtc > 0.005 else 'do NOT meaningfully'} add predictive value beyond odds alone.

  The top discriminating features are:
""")
for i, feat in enumerate(combined.head(10)['feature'].tolist(), 1):
    imp_val = importances.get(feat, 0)
    print(f"    {i:>2}. {feat:<40} (GB importance: {imp_val:.4f})")

print("\n" + "=" * 80)
