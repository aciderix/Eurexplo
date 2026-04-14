"""Ingest one day's PMU JSON into Supabase (races, participants).

Reads pmu_daily_json/pmu_<DDMMYYYY>.json (produced by pmu_scraper_turbo.py)
and upserts races + participants (with odds + result if available).

Usage:
  python backend/ingest_daily.py --date 14042026
  python backend/ingest_daily.py --date 14042026 --json pmu_daily_json/pmu_14042026.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

from supabase_client import get_client


def _upsert_ref(client, table: str, key_col: str, values: set[str]) -> dict[str, int]:
    """Upsert unique reference values, return mapping {name: id}."""
    values = {v for v in values if v}
    if not values:
        return {}
    rows = [{key_col: v} for v in values]
    client.table(table).upsert(rows, on_conflict=key_col).execute()
    existing = client.table(table).select(f"id,{key_col}").in_(key_col, list(values)).execute()
    return {r[key_col]: r["id"] for r in existing.data}


def ingest(date_str: str, json_path: str) -> None:
    """date_str format: DDMMYYYY (as produced by scraper)."""
    client = get_client()
    rows = json.loads(pathlib.Path(json_path).read_text())
    if not rows:
        print(f"No rows in {json_path}")
        return

    # Canonical date (YYYY-MM-DD)
    iso_date = f"{date_str[4:]}-{date_str[2:4]}-{date_str[0:2]}"

    # 1) Normalize reference tables
    hip_names      = {r["hippodrome"]  for r in rows if r.get("hippodrome")}
    horse_keys     = {r["nom"].upper().strip()    for r in rows if r.get("nom")}
    jockey_names   = {r["jockey"]      for r in rows if r.get("jockey")}
    trainer_names  = {r["entraineur"]  for r in rows if r.get("entraineur")}

    hip_map     = _upsert_ref(client, "hippodromes", "name", hip_names)
    horse_map   = _upsert_ref(client, "horses",      "horse_key", horse_keys)
    jockey_map  = _upsert_ref(client, "jockeys",     "name", jockey_names)
    trainer_map = _upsert_ref(client, "trainers",    "name", trainer_names)

    # 2) Group participants by race
    races_by_id: dict[str, dict[str, Any]] = {}
    participants_buf: list[dict[str, Any]] = []

    for r in rows:
        race_id = f"{date_str}_{r['reunion']}_{r['course']}"
        if race_id not in races_by_id:
            races_by_id[race_id] = {
                "race_id":       race_id,
                "date":          iso_date,
                "reunion":       r["reunion"],
                "course":        r["course"],
                "hippodrome_id": hip_map.get(r.get("hippodrome")),
                "discipline":    r.get("discipline"),
                "distance":      r.get("distance") or None,
                "terrain":       r.get("terrain"),
                "nb_partants":   r.get("nb_partants"),
            }
        participants_buf.append({
            "_race_id":          race_id,
            "num_pmu":           int(r["num_pmu"]) if r.get("num_pmu") else None,
            "_horse_key":        (r.get("nom") or "").upper().strip() or None,
            "_jockey_name":      r.get("jockey") or None,
            "_trainer_name":     r.get("entraineur") or None,
            "age":               r.get("age") or None,
            "musique":           r.get("musique"),
            "handicap_poids":    r.get("handicap_poids") or None,
            "deferre":           r.get("deferre"),
            "cote_probable":     r.get("cote_probable") or None,
            "cote_direct":       r.get("cote_direct") or None,
            "tendance":          r.get("tendance"),
            "finish_position":   r.get("finish_position") or None,
            "won":               bool(r.get("finish_position") == 1) if r.get("finish_position") else None,
            "placed":            bool(r.get("finish_position") and 1 <= r["finish_position"] <= 3) if r.get("finish_position") else None,
            "dividende_gagnant": r.get("dividende_gagnant") or None,
            "dividende_place":   r.get("dividende_place") or None,
        })

    # 3) Upsert races, get pk mapping
    client.table("races").upsert(list(races_by_id.values()), on_conflict="race_id").execute()
    race_pk = client.table("races").select("id,race_id").in_("race_id", list(races_by_id.keys())).execute()
    race_pk_map = {r["race_id"]: r["id"] for r in race_pk.data}

    # 4) Upsert participants
    parts_rows = []
    for p in participants_buf:
        parts_rows.append({
            "race_id":           race_pk_map[p.pop("_race_id")],
            "num_pmu":           p["num_pmu"],
            "horse_id":          horse_map.get(p.pop("_horse_key")),
            "jockey_id":         jockey_map.get(p.pop("_jockey_name")),
            "trainer_id":        trainer_map.get(p.pop("_trainer_name")),
            "age":               p["age"],
            "musique":           p["musique"],
            "handicap_poids":    p["handicap_poids"],
            "deferre":           p["deferre"],
            "cote_probable":     p["cote_probable"],
            "cote_direct":       p["cote_direct"],
            "tendance":          p["tendance"],
            "finish_position":   p["finish_position"],
            "won":               p["won"],
            "placed":            p["placed"],
            "dividende_gagnant": p["dividende_gagnant"],
            "dividende_place":   p["dividende_place"],
        })

    # Chunk inserts (PostgREST limits)
    CHUNK = 500
    for i in range(0, len(parts_rows), CHUNK):
        client.table("participants").upsert(
            parts_rows[i:i + CHUNK], on_conflict="race_id,num_pmu"
        ).execute()

    print(f"Ingested {iso_date}: {len(races_by_id)} races, {len(parts_rows)} participants")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True, help="DDMMYYYY (e.g. 14042026)")
    p.add_argument("--json", default=None, help="Path to pmu_<DDMMYYYY>.json")
    args = p.parse_args()
    path = args.json or f"pmu_daily_json/pmu_{args.date}.json"
    ingest(args.date, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
