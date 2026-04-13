#!/usr/bin/env python3
"""
PMU Turbo Scraper — ThreadPool avec multi-jours parallèles, 50 workers.
Traite plusieurs jours simultanément pour maximiser le débit.

Usage:
  python3 pmu_scraper_turbo.py                        # 2024-2026
  python3 pmu_scraper_turbo.py 01012014 13042026      # 2014-2026
  python3 pmu_scraper_turbo.py --resume               # Reprend

Output: pmu_full_dataset.csv
"""
import requests as req_lib
import json, csv, os, sys, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ── Config ────────────────────────────────────────────────────────────────────
BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}
MAX_WORKERS = 50           # 50 requêtes simultanées
BATCH_DAYS = 20            # 20 jours par batch
OUTPUT_CSV = "pmu_full_dataset.csv"
PROGRESS_FILE = ".pmu_scraper_progress.json"

CSV_FIELDS = [
    'date', 'reunion', 'course', 'hippodrome', 'discipline', 'distance',
    'terrain', 'nb_partants', 'num_pmu', 'nom', 'jockey', 'entraineur',
    'age', 'sexe', 'musique', 'nb_courses', 'nb_victoires', 'gains',
    'cote_probable', 'cote_direct', 'tendance',
    'finish_position', 'dividende_gagnant', 'dividende_place',
    'temps_km', 'deferre', 'handicap_poids',
]

# Thread-safe session
_session = req_lib.Session()
_session.headers.update(HEADERS)


def date_range(start_str, end_str):
    start = datetime.strptime(start_str, "%d%m%Y")
    end = datetime.strptime(end_str, "%d%m%Y")
    d = start
    while d <= end:
        yield d.strftime("%d%m%Y")
        d += timedelta(days=1)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'last_date': None, 'total_rows': 0, 'total_days': 0, 'done_dates': []}


def save_progress(data):
    data['done_dates'] = data.get('done_dates', [])[-500:]
    data['updated'] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(data, f)


def fetch_json(url):
    """Fetch JSON with retry."""
    for attempt in range(3):
        try:
            r = _session.get(url, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 404:
                return None
            elif r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except (req_lib.Timeout, req_lib.ConnectionError):
            if attempt < 2:
                time.sleep(1)
    return None


def fetch_race_data(date_str, r_num, c_num, meta):
    """Fetch participants + rapports for one race."""
    parts_data = fetch_json(f"{BASE}/{date_str}/R{r_num}/C{c_num}/participants")
    if not parts_data or not parts_data.get('participants'):
        return []

    participants = parts_data['participants']

    rapports = fetch_json(f"{BASE}/{date_str}/R{r_num}/C{c_num}/rapports-definitifs")

    gagnant_nums = {}
    place_nums = {}
    if rapports and isinstance(rapports, list):
        for rt in rapports:
            tp = rt.get('typePari', '')
            for rap in rt.get('rapports', []):
                comb = str(rap.get('combinaison', ''))
                div = rap.get('dividendePourUnEuro', 0)
                if tp == 'SIMPLE_GAGNANT':
                    gagnant_nums[comb] = div
                elif tp == 'SIMPLE_PLACE':
                    place_nums[comb] = div

    rows = []
    for p in participants:
        num = str(p.get('numPmu', ''))
        rap_direct = p.get('dernierRapportDirect', {}) or {}
        rap_probable = p.get('dernierRapportReference', {}) or {}

        cote_direct = rap_direct.get('rapport', 0)
        cote_probable = rap_probable.get('rapport', p.get('coteProbable', 0))
        tendance = rap_direct.get('tendance', '')

        ordreArrivee = p.get('ordreArrivee', 0)
        div_g = gagnant_nums.get(num, 0)
        div_p = place_nums.get(num, 0)

        if ordreArrivee:
            finish = ordreArrivee
        elif div_g and div_g > 0:
            finish = 1
        else:
            finish = 0

        gains_data = p.get('gainsParticipant', {})
        gains = gains_data.get('gainsCarriere', 0) if isinstance(gains_data, dict) else 0

        rows.append({
            'date': date_str,
            'reunion': f"R{r_num}",
            'course': f"C{c_num}",
            'hippodrome': meta.get('hippodrome', ''),
            'discipline': meta.get('discipline', ''),
            'distance': meta.get('distance', 0),
            'terrain': meta.get('terrain', ''),
            'nb_partants': len(participants),
            'num_pmu': num,
            'nom': p.get('nom', ''),
            'jockey': p.get('driver', p.get('jockey', '')),
            'entraineur': p.get('entraineur', ''),
            'age': p.get('age', ''),
            'sexe': p.get('sexe', ''),
            'musique': p.get('musique', ''),
            'nb_courses': p.get('nombreCourses', 0),
            'nb_victoires': p.get('nombreVictoires', 0),
            'gains': gains,
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
    """Get programme then submit all race fetches. Returns list of (r,c,meta) tasks."""
    prog = fetch_json(f"{BASE}/{date_str}")
    if not prog:
        return []

    programme = prog.get('programme', prog)
    tasks = []
    for reunion in programme.get('reunions', []):
        r_num = reunion.get('numOfficiel', 0)
        hip = reunion.get('hippodrome', {})
        hip_name = hip.get('libelleCourt', hip.get('libelleLong', '?'))
        for course in reunion.get('courses', []):
            c_num = course.get('numOrdre', course.get('numExterne', 0))
            meta = {
                'hippodrome': hip_name,
                'distance': course.get('distance', 0),
                'discipline': course.get('discipline', ''),
                'terrain': course.get('conditionPiste', ''),
            }
            tasks.append((date_str, r_num, c_num, meta))
    return tasks


def main():
    resume = '--resume' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    if len(args) >= 2:
        start_date, end_date = args[0], args[1]
    else:
        end = datetime.now()
        start = end - timedelta(days=730)
        start_date = start.strftime("%d%m%Y")
        end_date = end.strftime("%d%m%Y")

    dates = list(date_range(start_date, end_date))

    progress = load_progress() if resume else {'total_rows': 0, 'total_days': 0, 'done_dates': []}
    done_dates = set(progress.get('done_dates', []))

    if resume and done_dates:
        before = len(dates)
        dates = [d for d in dates if d not in done_dates]
        print(f"Reprise: {before - len(dates)} jours déjà faits, {progress['total_rows']} rows")

    total_rows = progress.get('total_rows', 0)
    total_days = progress.get('total_days', 0)

    # Init CSV
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    csv_file = open(OUTPUT_CSV, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not file_exists:
        writer.writeheader()
    csv_lock = Lock()

    print(f"=== PMU Turbo Scraper ({MAX_WORKERS} workers, {BATCH_DAYS} jours/batch) ===")
    print(f"Période: {start_date} → {end_date}")
    print(f"Jours restants: {len(dates)}")
    print()

    t0 = time.time()
    processed_days = 0

    try:
        # Process in batches of BATCH_DAYS
        for batch_start in range(0, len(dates), BATCH_DAYS):
            batch = dates[batch_start:batch_start + BATCH_DAYS]

            # Phase 1: Get all programmes for this batch (parallel)
            all_race_tasks = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                prog_futures = {executor.submit(scrape_day, d): d for d in batch}
                for future in as_completed(prog_futures):
                    try:
                        tasks = future.result()
                        all_race_tasks.extend(tasks)
                    except Exception:
                        pass

            # Phase 2: Fetch all participants + rapports (parallel, all races at once)
            batch_rows = []
            if all_race_tasks:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    race_futures = {
                        executor.submit(fetch_race_data, t[0], t[1], t[2], t[3]): t
                        for t in all_race_tasks
                    }
                    for future in as_completed(race_futures):
                        try:
                            rows = future.result()
                            if rows:
                                batch_rows.extend(rows)
                        except Exception:
                            pass

            # Write all rows from this batch
            if batch_rows:
                with csv_lock:
                    for row in batch_rows:
                        writer.writerow(row)
                    csv_file.flush()

            total_rows += len(batch_rows)
            processed_days += len(batch)
            for d in batch:
                done_dates.add(d)
                total_days += 1

            elapsed = time.time() - t0
            rate = processed_days / max(elapsed, 0.1) * 3600
            remaining_days = len(dates) - processed_days
            remaining_s = remaining_days / max(rate / 3600, 0.001)
            eta = f"{remaining_s/60:.0f}min" if remaining_s < 3600 else f"{remaining_s/3600:.1f}h"

            print(f"  [{processed_days}/{len(dates)}] {batch[0]}→{batch[-1]}: {len(all_race_tasks)} courses, +{len(batch_rows)} partants | Total: {total_rows} | {rate:.0f} j/h | ETA {eta}")

            # Save progress
            if processed_days % 100 < BATCH_DAYS:
                save_progress({
                    'last_date': batch[-1],
                    'total_rows': total_rows,
                    'total_days': total_days,
                    'done_dates': list(done_dates),
                })

    except KeyboardInterrupt:
        print(f"\n⏸ Interrompu.")
    finally:
        save_progress({
            'last_date': dates[-1] if dates else '',
            'total_rows': total_rows,
            'total_days': total_days,
            'done_dates': list(done_dates),
        })
        csv_file.close()

    elapsed = time.time() - t0
    print(f"\n=== Terminé ===")
    print(f"Total: {total_rows} partants, {total_days} jours")
    print(f"Durée: {elapsed/60:.1f} min ({elapsed/3600:.2f}h)")
    if os.path.exists(OUTPUT_CSV):
        print(f"Fichier: {OUTPUT_CSV} ({os.path.getsize(OUTPUT_CSV)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
