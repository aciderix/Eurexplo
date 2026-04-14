#!/usr/bin/env python3
"""
Charge tous les JSON journaliers de pmu_daily_json/ en un DataFrame unifié,
parse les champs utiles (date, musique), et sauvegarde en Parquet.

Output: pmu_dataset.parquet (~2.5M rows, tous les partants 2014-2026)
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

INPUT_DIR = "pmu_daily_json"
OUTPUT_PARQUET = "pmu_dataset.parquet"

# Musique format: "1a2h3p4s..." position + discipline code
MUSIQUE_RE = re.compile(r'(\d+|D|A|R|T)([aAhHpPsSmMcC])')


def parse_musique(music_str):
    """Parse 'DA5A3M2h' → list of (pos, discipline) tuples."""
    if not music_str:
        return []
    matches = MUSIQUE_RE.findall(music_str)
    out = []
    for pos, disc in matches:
        if pos == 'D':
            p = 0  # Disqualifié → position 0 (pire)
        elif pos in ('A', 'R', 'T'):
            p = 0  # Arrêté, Retiré, Tombé → 0
        else:
            try:
                p = int(pos)
            except ValueError:
                p = 0
        out.append((p, disc.lower()))
    return out


def load_one_file(path):
    """Load one daily JSON and return a list of dicts (one per runner)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []

    if not data:
        return []

    for r in data:
        # Parse date (DDMMYYYY → datetime)
        d = r.get('date', '')
        try:
            r['date_dt'] = datetime.strptime(d, "%d%m%Y")
        except ValueError:
            r['date_dt'] = pd.NaT

        # Parse musique → list of recent positions
        music_entries = parse_musique(r.get('musique', ''))
        r['music_parsed'] = music_entries
        r['music_len'] = len(music_entries)

        # Unique race_id
        r['race_id'] = f"{d}_{r.get('reunion', '')}_{r.get('course', '')}"

        # Runner_id (same horse across races)
        # We use name as proxy (PMU doesn't give horse_id stable across years in this dataset)
        r['horse_key'] = r.get('nom', '').strip().upper()

        # Win flag
        r['won'] = int(r.get('finish_position', 0) == 1)
        r['placed'] = int(r.get('finish_position', 0) in (1, 2, 3))

    return data


def main():
    files = sorted(os.listdir(INPUT_DIR))
    json_files = [f for f in files if f.endswith('.json')]
    print(f"Loading {len(json_files)} daily JSON files...")

    all_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(load_one_file, os.path.join(INPUT_DIR, f)): f for f in json_files}
        for i, fut in enumerate(as_completed(futures), 1):
            rows = fut.result()
            all_rows.extend(rows)
            if i % 500 == 0:
                print(f"  {i}/{len(json_files)} loaded, {len(all_rows):,} rows so far")

    print(f"\nTotal rows: {len(all_rows):,}")
    df = pd.DataFrame(all_rows)

    # Convert to proper types
    numeric_cols = ['distance', 'nb_partants', 'age', 'nb_courses', 'nb_victoires',
                    'gains', 'cote_probable', 'cote_direct', 'finish_position',
                    'dividende_gagnant', 'dividende_place', 'temps_km', 'handicap_poids']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Convert num_pmu to int safely
    df['num_pmu'] = pd.to_numeric(df['num_pmu'], errors='coerce')

    # Sort chronologically (critical for walk-forward)
    df = df.sort_values(['date_dt', 'reunion', 'course', 'num_pmu']).reset_index(drop=True)

    # Drop music_parsed (list column, serialization issue) — will recompute on load
    # Keep as JSON string
    df['musique_parsed_json'] = df['music_parsed'].apply(lambda x: json.dumps(x) if x else '[]')
    df = df.drop(columns=['music_parsed'])

    print(f"\nFinal shape: {df.shape}")
    print(f"Date range: {df['date_dt'].min()} → {df['date_dt'].max()}")
    print(f"Unique races: {df['race_id'].nunique():,}")
    print(f"Unique horses (by name): {df['horse_key'].nunique():,}")
    print(f"Unique jockeys: {df['jockey'].nunique():,}")
    print(f"Unique trainers: {df['entraineur'].nunique():,}")
    print(f"Win rate: {df['won'].mean():.3f}")

    # Basic sanity checks
    print(f"\nMissing cote_direct: {df['cote_direct'].isna().sum():,}")
    print(f"Missing cote_probable: {df['cote_probable'].isna().sum():,}")
    print(f"Missing finish_position: {df['finish_position'].isna().sum():,}")

    # Save as Parquet (compressed, fast reload)
    df.to_parquet(OUTPUT_PARQUET, compression='snappy', index=False)
    size_mb = os.path.getsize(OUTPUT_PARQUET) / 1e6
    print(f"\nSaved: {OUTPUT_PARQUET} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
