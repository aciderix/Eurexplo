#!/usr/bin/env python3
"""
PMU Parallel Scraper — Télécharge participants + résultats + cotes en parallèle.
Utilise la même logique que pmu-data-dl (20 requêtes simultanées).

Usage:
  python3 pmu_scraper_parallel.py                    # Scrape les 2 dernières années
  python3 pmu_scraper_parallel.py 01012024 31122025  # Plage de dates
  python3 pmu_scraper_parallel.py --resume            # Reprend là où on s'est arrêté

Output: pmu_full_dataset.csv (append mode, dédupliqué)
"""
import requests, json, csv, os, sys, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}
MAX_WORKERS = 20          # Requêtes parallèles (comme ton app)
PAUSE_BETWEEN_DAYS = 0.1  # Pause entre chaque jour (secondes)
OUTPUT_CSV = "pmu_full_dataset.csv"
PROGRESS_FILE = ".pmu_scraper_progress.json"
RETRY_MAX = 3
RETRY_DELAY = 2

CSV_FIELDS = [
    'date', 'reunion', 'course', 'hippodrome', 'discipline', 'distance',
    'terrain', 'nb_partants', 'num_pmu', 'nom', 'jockey', 'entraineur',
    'age', 'sexe', 'musique', 'nb_courses', 'nb_victoires', 'gains',
    'cote_probable', 'cote_direct', 'tendance',
    'finish_position', 'dividende_gagnant', 'dividende_place',
    'temps_km', 'deferre', 'handicap_poids',
]


def fetch_json(url, retries=RETRY_MAX):
    """Fetch JSON avec retry."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                return None
            elif r.status_code == 429:  # Rate limited
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            else:
                return None
        except (requests.Timeout, requests.ConnectionError):
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue
    return None


def get_programme(date_str):
    """Récupère le programme d'une journée → liste de (R, C, metadata)."""
    data = fetch_json(f"{BASE}/{date_str}")
    if not data:
        return []

    prog = data.get('programme', data)
    races = []
    for reunion in prog.get('reunions', []):
        r_num = reunion.get('numOfficiel', 0)
        hip = reunion.get('hippodrome', {})
        hip_name = hip.get('libelleCourt', hip.get('libelleLong', '?'))
        pays = reunion.get('pays', {}).get('code', 'FR')

        for course in reunion.get('courses', []):
            c_num = course.get('numOrdre', course.get('numExterne', 0))
            races.append({
                'r': r_num,
                'c': c_num,
                'hippodrome': hip_name,
                'distance': course.get('distance', 0),
                'discipline': course.get('discipline', ''),
                'terrain': course.get('conditionPiste', ''),
                'pays': pays,
            })
    return races


def fetch_race_data(date_str, race_meta):
    """Fetch participants + rapports pour une course. Retourne liste de rows."""
    r, c = race_meta['r'], race_meta['c']

    # Fetch participants
    parts_data = fetch_json(f"{BASE}/{date_str}/R{r}/C{c}/participants")
    if not parts_data or not parts_data.get('participants'):
        return []

    participants = parts_data['participants']

    # Fetch rapports définitifs (résultats)
    rapports = fetch_json(f"{BASE}/{date_str}/R{r}/C{c}/rapports-definitifs")

    # Parse résultats
    gagnant_nums = {}  # num -> dividende
    place_nums = {}    # num -> dividende
    if rapports and isinstance(rapports, list):
        for rapport_type in rapports:
            tp = rapport_type.get('typePari', '')
            for rap in rapport_type.get('rapports', []):
                comb = str(rap.get('combinaison', ''))
                div = rap.get('dividendePourUnEuro', 0)
                if tp == 'SIMPLE_GAGNANT':
                    gagnant_nums[comb] = div
                elif tp == 'SIMPLE_PLACE':
                    place_nums[comb] = div

    rows = []
    for p in participants:
        num = str(p.get('numPmu', ''))

        # Cotes
        rap_direct = p.get('dernierRapportDirect', {}) or {}
        rap_probable = p.get('dernierRapportReference', {}) or {}

        cote_direct = rap_direct.get('rapport', 0)
        cote_probable = rap_probable.get('rapport', p.get('coteProbable', 0))
        tendance = rap_direct.get('tendance', '')

        # Musique
        musique = p.get('musique', '')

        # Résultat
        # On détermine la position d'arrivée
        ordreArrivee = p.get('ordreArrivee', 0)

        # Dividendes
        div_g = gagnant_nums.get(num, 0)
        div_p = place_nums.get(num, 0)

        # Position (si pas dans ordreArrivee, on utilise le dividende)
        if ordreArrivee:
            finish = ordreArrivee
        elif div_g and div_g > 0:
            finish = 1
        else:
            finish = 0  # Inconnu

        rows.append({
            'date': date_str,
            'reunion': f"R{r}",
            'course': f"C{c}",
            'hippodrome': race_meta['hippodrome'],
            'discipline': race_meta.get('discipline', ''),
            'distance': race_meta.get('distance', 0),
            'terrain': race_meta.get('terrain', ''),
            'nb_partants': len(participants),
            'num_pmu': num,
            'nom': p.get('nom', ''),
            'jockey': p.get('driver', p.get('jockey', '')),
            'entraineur': p.get('entraineur', ''),
            'age': p.get('age', ''),
            'sexe': p.get('sexe', ''),
            'musique': musique,
            'nb_courses': p.get('nombreCourses', 0),
            'nb_victoires': p.get('nombreVictoires', 0),
            'gains': p.get('gainsParticipant', {}).get('gainsCarriere', 0) if isinstance(p.get('gainsParticipant'), dict) else 0,
            'cote_probable': cote_probable or 0,
            'cote_direct': cote_direct or 0,
            'tendance': tendance,
            'finish_position': finish,
            'dividende_gagnant': div_g,
            'dividende_place': div_p,
            'temps_km': p.get('tempsObtenu', 0),
            'deferre': p.get('deferre', ''),
            'handicap_poids': p.get('poidsConditionMonte', 0),
        })

    return rows


def scrape_day(date_str):
    """Scrape toutes les courses d'une journée en parallèle."""
    # 1. Récupérer le programme
    races = get_programme(date_str)
    if not races:
        return [], 0

    # 2. Fetch participants + résultats en parallèle
    all_rows = []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(races))) as executor:
        futures = {executor.submit(fetch_race_data, date_str, meta): meta for meta in races}
        for future in as_completed(futures):
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                pass

    return all_rows, len(races)


def load_progress():
    """Charge la progression pour reprendre."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'last_date': None, 'total_rows': 0, 'total_days': 0}


def save_progress(last_date, total_rows, total_days):
    """Sauvegarde la progression."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump({
            'last_date': last_date,
            'total_rows': total_rows,
            'total_days': total_days,
            'updated': datetime.now().isoformat()
        }, f)


def date_range(start_str, end_str):
    """Génère les dates DDMMYYYY entre start et end."""
    start = datetime.strptime(start_str, "%d%m%Y")
    end = datetime.strptime(end_str, "%d%m%Y")
    dates = []
    d = start
    while d <= end:
        dates.append(d.strftime("%d%m%Y"))
        d += timedelta(days=1)
    return dates


def main():
    # Parse arguments
    resume = '--resume' in sys.argv

    if len(sys.argv) >= 3 and not sys.argv[1].startswith('--'):
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        # Default: 2 dernières années
        end = datetime.now()
        start = end - timedelta(days=730)
        start_date = start.strftime("%d%m%Y")
        end_date = end.strftime("%d%m%Y")

    dates = date_range(start_date, end_date)

    # Resume support
    progress = load_progress()
    if resume and progress['last_date']:
        skip_until = progress['last_date']
        dates = [d for d in dates if d > skip_until or d == skip_until]
        dates = dates[1:]  # Skip the last completed day
        print(f"Reprise après {skip_until} ({progress['total_rows']} rows, {progress['total_days']} jours)")

    total_rows = progress.get('total_rows', 0) if resume else 0
    total_days = progress.get('total_days', 0) if resume else 0

    # Init CSV
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists or not resume:
        if not file_exists:
            writer.writeheader()

    print(f"=== PMU Parallel Scraper ===")
    print(f"Période: {start_date} → {end_date}")
    print(f"Jours à traiter: {len(dates)}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Output: {OUTPUT_CSV}")
    print()

    t0 = time.time()

    try:
        for i, date_str in enumerate(dates):
            rows, n_races = scrape_day(date_str)

            if rows:
                for row in rows:
                    writer.writerow(row)
                csv_file.flush()
                total_rows += len(rows)
                total_days += 1

            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1) * 3600  # jours/heure
            remaining = (len(dates) - i - 1) / max(rate / 3600, 0.001)

            status = f"[{i+1}/{len(dates)}] {date_str}: {n_races} courses, {len(rows)} partants"
            eta = f"ETA: {remaining/60:.0f}min" if remaining < 3600 else f"ETA: {remaining/3600:.1f}h"
            print(f"  {status} | Total: {total_rows} rows | {rate:.0f} jours/h | {eta}")

            # Save progress every 10 days
            if (i + 1) % 10 == 0:
                save_progress(date_str, total_rows, total_days)

            time.sleep(PAUSE_BETWEEN_DAYS)

    except KeyboardInterrupt:
        print(f"\n⏸ Interrompu. Progression sauvegardée.")
    finally:
        save_progress(date_str if dates else '', total_rows, total_days)
        csv_file.close()

    elapsed = time.time() - t0
    print(f"\n=== Terminé ===")
    print(f"Total: {total_rows} partants, {total_days} jours")
    print(f"Durée: {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
    print(f"Fichier: {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
