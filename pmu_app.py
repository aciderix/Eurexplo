#!/usr/bin/env python3
"""
PMU Predictor — App principale
Version: 1.0
Usage: python3 pmu_app.py
Input: course CSV (col: runner_id, race_id, morning_odds, final_odds, finish_position)
Output: predictions.json
"""

import csv, json, pickle, numpy as np
from collections import defaultdict

# ── Load model ─────────────────────────────────────────────────────────────────
model = pickle.load(open('model_gb.pkl','rb'))
with open('model_meta.json') as f: meta = json.load(f)
test_data = pickle.load(open('test_predictions.pkl','rb'))
races_test = test_data['races']
winner_map = {r['runner_id']: r['position']==1 for r in test_data['test']}

feature_names = meta['feature_names']

def build_features(r, race_horses):
    n = len(race_horses)
    odds_rank = sorted(race_horses, key=lambda x: x['morning_odds'])
    mr = next((i for i,x in enumerate(odds_rank) if x['runner_id']==r['runner_id']), n//2)
    fr_horse = sorted(race_horses, key=lambda x: x['final_odds'])
    fr = next((i for i,x in enumerate(fr_horse) if x['runner_id']==r['runner_id']), n//2)
    return np.array([r['final_odds'],r['morning_odds'],r['drift'],mr/max(n-1,1),fr/max(n-1,1),r['final_odds']/max(r['morning_odds'],0.01),1/max(r['final_odds'],0.01),n], dtype=float)

# ── Strategies ──────────────────────────────────────────────────────────────────
STRATEGIES = [
    {
        'id': 'HIGH_CONF',
        'name': 'Confiance haute',
        'color': '🟢',
        'description': ' prob >= 0.7 — winrate max, paris rares',
        'min_prob': 0.70,
        'odds_range': (1.5, 10.0),
        'field_range': (3, 25),
        'stake': 10.0,
    },
    {
        'id': 'MEDIUM_CONF',
        'name': 'Confiance medium',
        'color': '🟡',
        'description': ' prob >= 0.5 — équilibre winrate/volume',
        'min_prob': 0.50,
        'odds_range': (1.5, 8.0),
        'field_range': (4, 20),
        'stake': 5.0,
    },
    {
        'id': 'EDGE_ONLY',
        'name': 'Edge sur cotes (odds 2-5)',
        'color': '🔵',
        'description': ' prob >= 0.4 + cote 2-5 — volume max, edge stable',
        'min_prob': 0.40,
        'odds_range': (2.0, 5.0),
        'field_range': (4, 25),
        'stake': 3.0,
    },
]

def predict_race(race_id, race_horses):
    """Predict a single race. Returns dict with predictions and recommended bets."""
    for h in race_horses:
        feats = build_features(h, race_horses)
        h['_prob'] = model.predict_proba([feats])[0][1]
    
    # Morning favourite prob
    mf = min(race_horses, key=lambda x: x['morning_odds'])
    mf['_is_mf'] = True
    mf_prob = mf['_prob']
    
    # Drift analysis
    avg_drift = np.mean([h.get('drift',0) for h in race_horses])
    
    results = []
    for strat in STRATEGIES:
        bet = max(race_horses, key=lambda x: x['_prob'], default=None)
        if bet is None or bet['_prob'] < strat['min_prob']:
            results.append({'strategy': strat['id'], 'bet': None, 'prob': 0, 'edge': 0, 'verdict': 'SKIP'})
            continue
        
        mo = bet['morning_odds']
        fo = bet.get('final_odds', mo)
        implied = 1/mo
        edge = bet['_prob'] - implied
        
        # Breakeven WR for this odds
        breakeven = 1/mo
        
        # Kelly fraction (1/4 for safety)
        if edge > 0 and mo > 1:
            kelly = edge / (mo - 1)
            recommended_stake = min(strat['stake'], kelly * strat['stake'] * 4)  # cap at 4x base stake
        else:
            recommended_stake = 0
            edge = 0
        
        verdict = 'BET' if bet['_prob'] >= strat['min_prob'] and edge > 0 else 'SKIP'
        
        results.append({
            'strategy': strat['id'],
            'name': strat['name'],
            'color': strat['color'],
            'description': strat['description'],
            'bet_runner_id': bet.get('runner_id', ''),
            'bet_name': bet.get('horse_name', bet.get('runner_id','')),
            'prob': round(bet['_prob'], 3),
            'morning_odds': mo,
            'final_odds': fo,
            'drift': round(bet.get('drift',0), 4),
            'mf_prob': round(mf_prob, 3),
            'mf_prob_diff': round(bet['_prob'] - mf_prob, 3),
            'edge': round(edge, 4),
            'breakeven_wr': round(breakeven, 3),
            'recommended_stake': round(recommended_stake, 2),
            'verdict': verdict,
            'n_partants': len(race_horses),
            'avg_drift': round(avg_drift, 4),
        })
    
    return {
        'race_id': race_id,
        'n_partants': len(race_horses),
        'mf_prob': round(mf_prob, 3),
        'mf_name': mf.get('horse_name', mf.get('runner_id','')),
        'predictions': results,
        'drift_warning': '⚠️ DRIFT HIGH' if avg_drift > 0.05 else ('⚠️ DRIFT LOW' if avg_drift < -0.03 else 'OK'),
    }

def run_backtest():
    """Backtest all strategies on held-out test set."""
    winner_map = {r['runner_id']: r['position']==1 for r in test_data['test']}
    
    by_strat = defaultdict(lambda: {'bets': [], 'wins': 0, 'pnl': 0, 'race_ids': set()})
    
    for race_id, race_horses in races_test.items():
        if len(race_horses) < 4: continue
        for h in race_horses:
            feats = build_features(h, race_horses)
            h['_prob'] = model.predict_proba([feats])[0][1]
        mf = min(race_horses, key=lambda x: x['morning_odds'])
        mf_prob = mf['_prob']
        avg_drift = np.mean([h.get('drift',0) for h in race_horses])
        
        for strat in STRATEGIES:
            bet = max(race_horses, key=lambda x: x['_prob'], default=None)
            if bet is None or bet['_prob'] < strat['min_prob']:
                continue
            
            mo = bet['morning_odds']
            fo = bet.get('final_odds', mo)
            implied = 1/mo
            edge = bet['_prob'] - implied
            
            if edge <= 0: continue
            
            kelly = edge / (mo - 1)
            stake = min(strat['stake'], kelly * strat['stake'] * 4)
            win = winner_map.get(bet['runner_id'], False)
            pnl = (fo * stake - stake) if win else -stake
            
            by_strat[strat['id']]['bets'].append({
                'race_id': race_id,
                'runner_id': bet['runner_id'],
                'odds': mo,
                'final_odds': fo,
                'prob': bet['_prob'],
                'edge': edge,
                'stake': stake,
                'win': win,
                'pnl': pnl,
                'n_partants': len(race_horses),
            })
    
    print("\n" + "="*80)
    print("BACKTEST COMPLET — STRATÉGIES PMU")
    print("="*80)
    
    for strat in STRATEGIES:
        s = by_strat[strat['id']]
        if not s['bets']: continue
        
        n = len(s['bets'])
        wins = sum(1 for b in s['bets'] if b['win'])
        wr = wins/n
        pnl = sum(b['pnl'] for b in s['bets'])
        avg_odds = np.mean([b['odds'] for b in s['bets']])
        avg_edge = np.mean([b['edge'] for b in s['bets']])
        roi = pnl / (n * strat['stake']) * 100
        bets_per_month = n / (len(set(b['race_id'][:8] for b in s['bets'])) / 4) if s['bets'] else 0
        
        print(f"\n{strat['color']} {strat['name']} [{strat['id']}]")
        print(f"  Paris joués     : {n}")
        print(f"  Win rate        : {wr:.1%}")
        print(f"  Cote moyenne    : {avg_odds:.2f}")
        print(f"  Edge moyen      : +{avg_edge:.1%}")
        print(f"  P&L total       : {pnl:+.2f} EUR")
        print(f"  ROI (mise {strat['stake']} EUR): {roi:+.1f}%")
        print(f"  Paris/mois est. : ~{bets_per_month:.0f}")
        print(f"  Exposition/jour : ~{bets_per_month/30*strat['stake']:.1f} EUR")
        
        # Monthly breakdown
        s['bets'].sort(key=lambda x: x['race_id'][:8])
        months = defaultdict(lambda: {'bets':0,'wins':0,'pnl':0,'stakes':0})
        for b in s['bets']:
            m = b['race_id'][:6]
            months[m]['bets'] += 1
            months[m]['wins'] += 1 if b['win'] else 0
            months[m]['pnl'] += b['pnl']
            months[m]['stakes'] += b['stake']
        
        print(f"  mois     | n   | WR    | P&L    | ROI")
        print(f"  " + "-"*48)
        for m in sorted(months.keys())[-12:]:
            d = months[m]
            wr_m = d['wins']/d['bets'] if d['bets'] else 0
            roi_m = d['pnl']/(d['stakes']+0.001)*100
            print(f"  {m} | {d['bets']:>3} | {wr_m:>5.1%} | {d['pnl']:+6.1f} | {roi_m:+6.1f}%")
        
        # Worst month
        worst = min(months.items(), key=lambda kv: kv[1]['pnl'])
        print(f"  Pire mois : {worst[0]} — {worst[1]['pnl']:.2f} EUR")
    
    # Grand total
    all_bets = []
    for strat in STRATEGIES:
        all_bets.extend(by_strat[strat['id']]['bets'])
    
    print(f"\n{'='*80}")
    total_pnl = sum(b['pnl'] for b in all_bets)
    total_stake = sum(b['stake'] for b in all_bets)
    print(f"S Total: {len(all_bets)} paris, P&L={total_pnl:+.2f} EUR, ROI={total_pnl/total_stake*100:+.1f}%")
    print(f"\nModel: GradientBoosting n_estimators=300 max_depth=5")
    print(f"Features: {feature_names}")
    print(f"Train: {meta['train_n']} runners ({meta['train_dates'][0]} -> {meta['train_dates'][1]})")
    print(f"Test:  {meta['test_n']} runners ({meta['test_dates'][0]} -> {meta['test_dates'][1]})")

def predict_from_csv(csv_path):
    """Predict from a CSV file of upcoming races."""
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f): rows.append(row)
    
    races = defaultdict(list)
    for r in rows:
        try:
            r['morning_odds'] = float(r.get('morning_odds',0))
            r['final_odds'] = float(r.get('final_odds', r['morning_odds']))
            if r['morning_odds'] < 1.1: continue
            r['drift'] = (r['final_odds'] - r['morning_odds']) / r['morning_odds']
        except: continue
        races[r['race_id']].append(r)
    
    results = []
    for race_id, race_horses in sorted(races.items()):
        if len(race_horses) < 4: continue
        res = predict_race(race_id, race_horses)
        results.append(res)
    
    return results

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        # Predict mode
        results = predict_from_csv(sys.argv[1])
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        # Backtest mode
        run_backtest()