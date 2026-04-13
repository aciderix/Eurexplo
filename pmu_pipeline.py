#!/usr/bin/env python3
"""
PMU Complete Pipeline — scrape participants + résultats pour construire dataset ML.
Usage: python3 pmu_pipeline.py [date]
"""
import requests, json, time, sys, csv, os
from datetime import datetime, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SLEEP = 0.4

def fetch(url, timeout=10):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    return r.json() if r.status_code == 200 else None

def discover_races(date):
    """Trouve toutes les courses d'une journée."""
    races = {}
    for r in range(1, 6):
        for c in range(1, 15):
            url = f"https://offline.turfinfo.api.pmu.fr/rest/client/1/programme/{date}/R{r}/C{c}/participants"
            data = fetch(url)
            if data and data.get('participants'):
                races[f"R{r}C{c}"] = data['participants']
            time.sleep(SLEEP)
    return races

def fetch_results(date, r_num, c_num):
    """Fetch les rapports définitifs."""
    url = f"https://offline.turfinfo.api.pmu.fr/rest/client/1/programme/{date}/R{r_num}/C{c_num}/rapports-definitifs"
    return fetch(url)

def parse_rapports(rapports_list):
    """Parse les rapports pour extraire les numéros gagnants."""
    winners = {}
    for rapport_type in rapports_list:
        ptype = rapport_type.get('typePari', '')
        for rap in rapport_type.get('rapports', []):
            comb = rap.get('combinaison', '')
            divid = rap.get('dividendePourUnEuro', 0)
            winners[f"{ptype}_{comb}"] = divid
    return winners

def build_row(p, race_key, date, winners=None):
    """Extrait les features d'un partant."""
    rap = p.get('dernierRapportDirect', {})
    # Extraire la cote (peut être float ou int)
    cote_raw = rap.get('rapport', 0)
    try:
        cote = float(cote_raw)
    except:
        cote = 0
    
    # Tendance: D=down, U=up, ==stable
    tendance = rap.get('tendance', '=')
    
    # Musique: 1h2h3h... 
    music = p.get('libelleArtiste', '')
    
    # Recent form: dernières places
    places = []
    for m in music.split()[:6]:
        if m and m[0].isdigit():
            try:
                places.append(int(m[0]))
            except:
                pass
    
    recent_avg = sum(places)/len(places) if places else 0
    recent_min = min(places) if places else 0
    nb_recent = len(places)
    
    # Victoires dans la musique
    nb_wins_music = sum(1 for p in places if p == 1)
    
    # Type de course (h=obstacle, p=plat, a=attelé)
    types = [m[1] if len(m)>1 else '' for m in music.split()[:6] if m]
    discipline_counts = {}
    for t in types:
        discipline_counts[t] = discipline_counts.get(t, 0) + 1
    main_discipline = max(discipline_counts, key=discipline_counts.get) if discipline_counts else ''
    
    # Gains (en centimes d'euro)
    gains = p.get('gains', 0) or 0
    
    # Nombre de courses
    nb_courses = p.get('nombreCourses', 0) or 0
    nb_victoires = p.get('nombreVictoires', 0) or 0
    win_rate = nb_victoires / nb_courses if nb_courses > 0 else 0
    
    row = {
        'date': date,
        'race': race_key,
        'discipline': p.get('discipline', ''),
        'num': p.get('numPmu', ''),
        'id_pmu': p.get('idPmu', ''),
        'nom': p.get('nom', ''),
        'cote_pmu': cote,
        'tendance': tendance,
        'music': music,
        'jockey': p.get('nomJockey', ''),
        'entraineur': p.get('entraineur', ''),
        'age': p.get('age', ''),
        'sexe': p.get('sexe', ''),
        'deferre': p.get('deferre', ''),
        'gains': gains,
        'nb_courses': nb_courses,
        'nb_victoires': nb_victoires,
        'win_rate': round(win_rate, 4),
        'recent_avg': round(recent_avg, 3),
        'recent_min': recent_min,
        'nb_recent': nb_recent,
        'nb_wins_music': nb_wins_music,
        'main_discipline': main_discipline,
    }
    
    # Résultats (si disponibles)
    if winners:
        # Trouver si ce cheval était dans les rapports gagnants
        num = p.get('numPmu', '')
        
        # Simple Gagnant: trouve le numéro gagnant
        sg = next((x for x in winners if x.get('typePari') == 'SIMPLE_GAGNANT'), None)
        if sg:
            winning_nums = [r.get('combinaison') for r in sg.get('rapports', [])]
            row['won_simple_gagnant'] = 1 if num in winning_nums else 0
            div = next((r.get('dividendePourUnEuro') for r in sg.get('rapports', []) if r.get('combinaison') == num), 0)
            row['dividende_sg'] = div
        
        # Simple Placé: cheval dans les placés?
        sp = next((x for x in winners if x.get('typePari') == 'SIMPLE_PLACE'), None)
        if sp:
            placed_nums = [r.get('combinaison') for r in sp.get('rapports', [])]
            row['won_simple_place'] = 1 if num in placed_nums else 0
        
        # 2sur4
        ds4 = next((x for x in winners if x.get('typePari') == 'DEUX_SUR_QUATRE'), None)
        if ds4:
            combo = ds4.get('rapports', [{}])[0].get('combinaison', '')
            parts = combo.split('-')
            if len(parts) >= 2:
                row['in_2sur4_1'] = parts[0]
                row['in_2sur4_2'] = parts[1]
    else:
        row['won_simple_gagnant'] = -1
        row['dividende_sg'] = -1
        row['won_simple_place'] = -1
    
    return row

def scrape_day(date):
    """Scrape participants + résultats pour une journée."""
    print(f"\n=== {date} ===")
    
    # Découvrir les courses
    races = discover_races(date)
    print(f"  {len(races)} courses")
    if not races:
        return []
    
    rows = []
    for race_key, participants in races.items():
        r_num = race_key[1]
        c_num = race_key[3:]
        
        # Résultats
        results = fetch_results(date, r_num, c_num)
        time.sleep(SLEEP)
        
        for p in participants:
            row = build_row(p, race_key, date, results)
            rows.append(row)
        
        print(f"  {race_key}: {len(participants)} partants" + (" + résultats" if results else ""))
        time.sleep(SLEEP)
    
    return rows

def load_existing(path):
    existing = set()
    if os.path.exists(path):
        with open(path) as f:
            header = f.readline().strip().split(',')
            for line in f:
                parts = line.strip().split(',')
                if parts and len(parts) >= 3:
                    existing.add(f"{parts[0]},{parts[1]},{parts[5]}")  # date,race,nom
    return existing

def save_rows(rows, csv_path):
    header = 'date,race,discipline,num,id_pmu,nom,cote_pmu,tendance,music,jockey,entraineur,age,sexe,deferre,gains,nb_courses,nb_victoires,win_rate,recent_avg,recent_min,nb_recent,nb_wins_music,main_discipline,won_simple_gagnant,dividende_sg,won_simple_place\n'
    
    existing = load_existing(csv_path)
    
    with open(csv_path, 'a') as f:
        if not existing:
            f.write(header)
        for r in rows:
            key = f"{r['date']},{r['race']},{r['nom']}"
            if key not in existing:
                f.write(f"{r['date']},{r['race']},{r['discipline']},{r['num']},{r['id_pmu']},{r['nom']},{r['cote_pmu']},{r['tendance']},{r['music']},{r['jockey']},{r['entraineur']},{r['age']},{r['sexe']},{r['deferre']},{r['gains']},{r['nb_courses']},{r['nb_victoires']},{r['win_rate']},{r['recent_avg']},{r['recent_min']},{r['nb_recent']},{r['nb_wins_music']},{r['main_discipline']},{r.get('won_simple_gagnant',-1)},{r.get('dividende_sg',-1)},{r.get('won_simple_place',-1)}\n")

if __name__ == '__main__':
    CSV = '/home/workspace/Eurexplo/pmu_data.csv'
    
    if len(sys.argv) > 1:
        # Scrape une date précise
        date = sys.argv[1]
        rows = scrape_day(date)
        if rows:
            save_rows(rows, CSV)
            print(f"Sauvegardé {len(rows)} partants dans {CSV}")
    else:
        # Scrape les derniers dimanches (jours principaux)
        dates = []
        d = datetime.now()
        for i in range(60):
            d2 = d - timedelta(days=i)
            # Sunday = main days, also Wednesday sometimes
            if d2.weekday() == 6:  # Sunday
                dates.append(d2.strftime("%d%m%Y"))
        
        print(f"Scraping {len(dates)} dimanches...")
        for date in dates[:5]:  # Limit to avoid rate limiting
            try:
                rows = scrape_day(date)
                if rows:
                    save_rows(rows, CSV)
                    print(f"  → {len(rows)} ajoutés")
            except Exception as e:
                print(f"  Erreur {date}: {e}")
            time.sleep(5)  # Pause entre les jours
        
        print(f"\nDataset complet dans {CSV}")
        
        # Stats
        total = 0
        with_result = 0
        won = 0
        try:
            with open(CSV) as f:
                f.readline()
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 24:
                        total += 1
                        wsg = parts[23]
                        if wsg == '1':
                            with_result += 1
                            won += 1
                        elif wsg == '0':
                            with_result += 1
        except:
            pass
        print(f"Lignes: {total}, avec résultats: {with_result}, gagnants SG: {won} ({100*won/with_result:.1f}% WR)" if with_result else "")