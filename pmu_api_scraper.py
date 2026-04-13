#!/usr/bin/env python3
"""
PMU API Scraper — récupère automatiquement cotes, musique, jockeys, form.
Usage: python3 pmu_api_scraper.py [date JJMMAAAA]
"""
import requests, json, time, sys
from datetime import datetime, timedelta

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
SLEEP = 0.5  # Entre chaque requête pour éviter le ban

def fetch_race(date, r_num, c_num):
    """Fetch participants d'une course."""
    url = f"https://offline.turfinfo.api.pmu.fr/rest/client/1/programme/{date}/R{r_num}/C{c_num}/participants"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    return r.json()

def discover_races(date):
    """Trouve toutes les courses d'une journée (probe systématique)."""
    races = {}
    # journée classique: R1-R5, C1-C12
    for r in range(1, 6):
        for c in range(1, 15):
            data = fetch_race(date, r, c)
            if data and data.get('participants'):
                n = len(data['participants'])
                races[f"R{r}C{c}"] = {
                    'reunion': r, 'course': c,
                    'partants': data['participants'],
                    'n_partants': n
                }
                print(f"  R{r}C{c}: {n} partants")
            time.sleep(SLEEP)
    return races

def extract_participants(races, date):
    """Extrait les données de chaque partant."""
    rows = []
    for race_key, race_data in races.items():
        r = race_data['reunion']
        c = race_data['course']
        for p in race_data['partants']:
            rap = p.get('dernierRapportDirect', {})
            rows.append({
                'date': date,
                'reunion': race_key,
                'discipline': p.get('discipline', ''),
                'num': p.get('numPmu', ''),
                'id_pmu': p.get('idPmu', ''),
                'nom': p.get('nom', ''),
                'cote_pmu': rap.get('rapport'),
                'tendance': rap.get('tendance'),  # = stable, D down, U up
                'music': p.get('libelleArtiste', ''),
                'jockey': p.get('nomJockey', ''),
                'entraineur': p.get('entraineur', ''),
                'age': p.get('age', ''),
                'sexe': p.get('sexe', ''),
                'deferre': p.get('deferre', ''),
                'gains': p.get('gains', 0),
                'carriere': p.get('carriere', ''),
                'courses_victoires': p.get('nombreCourses', 0),
                'victoires': p.get('nombreVictoires', 0),
                'handicap': p.get('handicap', ''),
            })
    return rows

def fetch_results(date, r_num, c_num):
    """Fetch les rapports definitifs (résultats)."""
    url = f"https://offline.turfinfo.api.pmu.fr/rest/client/1/programme/{date}/R{r_num}/C{c_num}/rapports-definitifs"
    r = requests.get(url, headers=HEADERS, timeout=10)
    if r.status_code != 200:
        return None
    return r.json()

def load_existing_csv(csv_path):
    """Charge les lignes existantes pour éviter les doublons."""
    existing = set()
    try:
        with open(csv_path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if parts:
                    existing.add(f"{parts[0]},{parts[3]},{parts[5]}")  # date,reunion,nom
    except:
        pass
    return existing

def save_to_csv(rows, csv_path):
    """Sauve en CSV."""
    existing = load_existing_csv(csv_path)
    with open(csv_path, 'a') as f:
        header = 'date,reunion,discipline,num,id_pmu,nom,cote_pmu,tendance,music,jockey,entraineur,age,sexe,deferre,gains,carriere,nb_courses,victoires\n'
        if existing:
            f.write(header)
        for r in rows:
            key = f"{r['date']},{r['reunion']},{r['nom']}"
            if key not in existing:
                f.write(f"{r['date']},{r['reunion']},{r['discipline']},{r['num']},{r['id_pmu']},{r['nom']},{r['cote_pmu']},{r.get('tendance','')},{r['music']},{r['jockey']},{r['entraineur']},{r['age']},{r['sexe']},{r['deferre']},{r['gains']},{r['carriere']},{r['courses_victoires']},{r['victoires']}\n")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = datetime.now().strftime("%d%m%Y")
    
    print(f"=== PMU API Scraper — {date} ===")
    
    # Découverte des courses
    print("Découverte des courses...")
    races = discover_races(date)
    print(f"\n{len(races)} courses trouvées")
    
    if not races:
        print("Aucune course. Date inactive ou erreur.")
        print("Essayer une date passée: python3 pmu_api_scraper.py 12042026")
        exit(1)
    
    # Extraction
    print("\nExtraction des données...")
    rows = extract_participants(races, date)
    print(f"{len(rows)} partants extraits")
    
    # Sauvegarde JSON
    with open(f'/tmp/pmu_data_{date}.json', 'w') as f:
        json.dump({'date': date, 'races': races, 'partants': rows}, f, indent=2, ensure_ascii=False)
    print(f"Sauvegardé dans /tmp/pmu_data_{date}.json")
    
    # CSV pour ML
    save_to_csv(rows, '/tmp/pmu_data.csv')
    print(f"Appendé dans /tmp/pmu_data.csv")