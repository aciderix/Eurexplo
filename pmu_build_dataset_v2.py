#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMU Dataset Builder V2 -- exhaustive flatten of pmu_data/*.json.gz into parquet.

Produces two parquet files:
  - pmu_dataset_v2.parquet      : one row per (race, participant)   ~2.5M rows
  - pmu_course_raw_v2.parquet   : one row per race, with nested JSON blobs
                                  (performances_detaillees, masse_enjeu,
                                   rapports_definitifs, pronostics_detailles,
                                   pronostics, citations) serialized as strings

Usage:
  python pmu_build_dataset_v2.py                       # full dataset
  python pmu_build_dataset_v2.py --from 2020-01-01     # partial
  python pmu_build_dataset_v2.py --workers 4

Output columns are deliberately wide (~100 scalar columns) so a downstream
feature-engineering step can exploit every signal, not just what pros look at.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = Path("pmu_data")
OUT_PARTICIPANTS = Path("pmu_dataset_v2.parquet")
OUT_COURSES = Path("pmu_course_raw_v2.parquet")


# ── Participant schema (scalar columns) ──────────────────────────────────────

PARTICIPANT_SCHEMA = pa.schema([
    # ids
    ("race_id",                pa.string()),
    ("file_date",              pa.string()),           # YYYY-MM-DD
    ("reunion_num",            pa.int32()),
    ("course_num",             pa.int32()),
    ("num_pmu",                pa.int32()),

    # reunion
    ("hippodrome_code",        pa.string()),
    ("hippodrome_libelle",     pa.string()),
    ("pays_code",              pa.string()),
    ("nature_reunion",         pa.string()),
    ("date_reunion_ts",        pa.int64()),            # ms since epoch

    # course
    ("libelle",                pa.string()),
    ("libelle_court",          pa.string()),
    ("discipline",             pa.string()),
    ("specialite",             pa.string()),
    ("categorie_particuliere", pa.string()),
    ("condition_sexe",         pa.string()),
    ("distance",               pa.int32()),
    ("distance_unit",          pa.string()),
    ("corde",                  pa.string()),
    ("parcours",               pa.string()),
    ("nb_declares_partants",   pa.int32()),
    ("nb_partants_reels",      pa.int32()),
    ("heure_depart_ts",        pa.int64()),
    ("grand_prix_national_trot", pa.bool_()),
    ("pari_multi_courses",     pa.bool_()),
    ("pari_special",           pa.bool_()),
    ("montant_prix",           pa.int64()),
    ("montant_total_offert",   pa.int64()),
    ("montant_offert_1er",     pa.int64()),
    ("montant_offert_2eme",    pa.int64()),
    ("montant_offert_3eme",    pa.int64()),
    ("montant_offert_4eme",    pa.int64()),
    ("montant_offert_5eme",    pa.int64()),
    ("conditions_text",        pa.string()),
    ("course_trackee",         pa.bool_()),
    ("statut_course",          pa.string()),
    ("arrivee_definitive",     pa.bool_()),

    # participant -- core
    ("nom",                    pa.string()),
    ("age",                    pa.int32()),
    ("sexe",                   pa.string()),
    ("race",                   pa.string()),
    ("statut_participant",     pa.string()),
    ("oeilleres",              pa.string()),
    ("proprietaire",           pa.string()),
    ("entraineur",             pa.string()),
    ("driver",                 pa.string()),
    ("driver_change",          pa.bool_()),
    ("indicateur_inedit",      pa.bool_()),
    ("nombre_courses",         pa.int32()),
    ("nombre_victoires",       pa.int32()),
    ("nombre_places",          pa.int32()),
    ("nombre_places_second",   pa.int32()),
    ("nombre_places_troisieme", pa.int32()),
    ("nom_pere",               pa.string()),
    ("nom_mere",               pa.string()),
    ("nom_pere_mere",          pa.string()),
    ("jument_pleine",          pa.bool_()),
    ("engagement",             pa.bool_()),
    ("supplement",             pa.int64()),
    ("poids_condition_monte_change", pa.bool_()),
    ("allure",                 pa.string()),
    ("musique",                pa.string()),
    ("handicap_distance",      pa.int32()),
    ("handicap_poids",         pa.int32()),
    ("handicap_valeur",        pa.int32()),
    ("place_corde",            pa.int32()),
    ("eleveur",                pa.string()),
    ("temps_obtenu",           pa.int64()),
    ("reduction_kilometrique", pa.int64()),
    ("distance_cheval_precedent", pa.int32()),
    ("pays_entrainement",      pa.string()),
    ("avis_entraineur",        pa.string()),
    ("url_casaque",            pa.string()),

    # participant -- derived (target + market)
    ("finish_position",        pa.int32()),
    ("won",                    pa.bool_()),
    ("placed",                 pa.bool_()),

    # gainsParticipant.*
    ("gains_annee_en_cours",   pa.int64()),
    ("gains_annee_precedente", pa.int64()),
    ("gains_carriere",         pa.int64()),
    ("gains_place",            pa.int64()),
    ("gains_victoires",        pa.int64()),

    # robe.*
    ("robe_code",              pa.string()),
    ("robe_libelle_court",     pa.string()),
    ("robe_libelle_long",      pa.string()),

    # dernierRapportDirect.* (live odds)
    ("drd_rapport",            pa.float64()),
    ("drd_favoris",            pa.bool_()),
    ("drd_grosse_prise",       pa.bool_()),
    ("drd_nombre_indicateur_tendance", pa.int32()),
    ("drd_permutation",        pa.int32()),
    ("drd_type_pari",          pa.string()),
    ("drd_type_rapport",       pa.string()),
    ("drd_date_rapport_ts",    pa.int64()),

    # dernierRapportReference.* (probable odds reference)
    ("drr_rapport",            pa.float64()),
    ("drr_favoris",            pa.bool_()),
    ("drr_grosse_prise",       pa.bool_()),
    ("drr_nombre_indicateur_tendance", pa.int32()),
    ("drr_permutation",        pa.int32()),
    ("drr_type_pari",          pa.string()),
    ("drr_type_rapport",       pa.string()),
    ("drr_date_rapport_ts",    pa.int64()),
])


# ── Course-level raw schema (JSON blobs) ─────────────────────────────────────

COURSE_SCHEMA = pa.schema([
    ("race_id",                 pa.string()),
    ("file_date",               pa.string()),
    ("reunion_num",             pa.int32()),
    ("course_num",              pa.int32()),
    ("ordre_arrivee_json",      pa.string()),
    ("performances_detaillees_json", pa.string()),
    ("masse_enjeu_json",        pa.string()),
    ("rapports_definitifs_json", pa.string()),
    ("pronostics_json",         pa.string()),
    ("pronostics_detailles_json", pa.string()),
    ("citations_json",          pa.string()),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _as_int(v: Any) -> int | None:
    if v is None or isinstance(v, bool):
        return None if v is None else int(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    return bool(v) if v in (0, 1) else None


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return None
    return str(v)


def _json_or_none(v: Any) -> str | None:
    if v is None:
        return None
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _extract_file_date(path: Path) -> str:
    # "2024-06-15.json.gz" -> "2024-06-15"
    return path.name.split(".json")[0]


def _finish_position(order: Any) -> int | None:
    """participant.ordreArrivee is the finish rank (int) or None/0 for DNF."""
    v = _as_int(order)
    if v is None or v == 0:
        return None
    return v


def _flatten_file(path_str: str) -> tuple[list[dict], list[dict], int]:
    """Parse one .json.gz file and return (participant_rows, course_rows, err_count)."""
    path = Path(path_str)
    part_rows: list[dict] = []
    course_rows: list[dict] = []
    errors = 0

    try:
        with gzip.open(path, "rb") as f:
            day_data = json.load(f)
    except Exception:
        return part_rows, course_rows, 1

    if not day_data:
        return part_rows, course_rows, 0

    file_date = _extract_file_date(path)

    for reunion_obj in day_data:
        r_num = _as_int(reunion_obj.get("reunion"))
        info_r = reunion_obj.get("info_reunion") or {}
        hippo = info_r.get("hippodrome") or {}
        pays = info_r.get("pays") or {}
        hippodrome_code = _as_str(hippo.get("code") or hippo.get("codeHippodrome"))
        hippodrome_libelle = _as_str(hippo.get("libelleCourt") or hippo.get("libelleLong"))
        pays_code = _as_str(pays.get("code"))
        nature_reunion = _as_str(info_r.get("nature"))
        date_reunion_ts = _as_int(info_r.get("dateReunion"))

        for course_obj in reunion_obj.get("courses", []) or []:
            c_num = _as_int(course_obj.get("course"))
            ic = course_obj.get("info_course") or {}
            c_hippo = ic.get("hippodrome") or {}

            # fallback hippodrome from course-level
            if not hippodrome_code:
                hippodrome_code = _as_str(c_hippo.get("codeHippodrome"))
            if not hippodrome_libelle:
                hippodrome_libelle = _as_str(c_hippo.get("libelleCourt"))

            race_id = f"{file_date}_R{r_num}_C{c_num}"

            parts_container = course_obj.get("participants") or {}
            pl = parts_container.get("participants") or []
            nb_partants_reels = len(pl)

            # Course-level common dict
            course_common = {
                "race_id":                 race_id,
                "file_date":               file_date,
                "reunion_num":             r_num,
                "course_num":              c_num,
                "hippodrome_code":         hippodrome_code,
                "hippodrome_libelle":      hippodrome_libelle,
                "pays_code":               pays_code,
                "nature_reunion":          nature_reunion,
                "date_reunion_ts":         date_reunion_ts,
                "libelle":                 _as_str(ic.get("libelle")),
                "libelle_court":           _as_str(ic.get("libelleCourt")),
                "discipline":              _as_str(ic.get("discipline")),
                "specialite":              _as_str(ic.get("specialite")),
                "categorie_particuliere":  _as_str(ic.get("categorieParticularite")),
                "condition_sexe":          _as_str(ic.get("conditionSexe")),
                "distance":                _as_int(ic.get("distance")),
                "distance_unit":           _as_str(ic.get("distanceUnit")),
                "corde":                   _as_str(ic.get("corde")),
                "parcours":                _as_str(ic.get("parcours")),
                "nb_declares_partants":    _as_int(ic.get("nombreDeclaresPartants")),
                "nb_partants_reels":       nb_partants_reels,
                "heure_depart_ts":         _as_int(ic.get("heureDepart")),
                "grand_prix_national_trot": _as_bool(ic.get("grandPrixNationalTrot")),
                "pari_multi_courses":      _as_bool(ic.get("pariMultiCourses")),
                "pari_special":            _as_bool(ic.get("pariSpecial")),
                "montant_prix":            _as_int(ic.get("montantPrix")),
                "montant_total_offert":    _as_int(ic.get("montantTotalOffert")),
                "montant_offert_1er":      _as_int(ic.get("montantOffert1er")),
                "montant_offert_2eme":     _as_int(ic.get("montantOffert2eme")),
                "montant_offert_3eme":     _as_int(ic.get("montantOffert3eme")),
                "montant_offert_4eme":     _as_int(ic.get("montantOffert4eme")),
                "montant_offert_5eme":     _as_int(ic.get("montantOffert5eme")),
                "conditions_text":         _as_str(ic.get("conditions")),
                "course_trackee":          _as_bool(ic.get("courseTrackee")),
                "statut_course":           _as_str(ic.get("statut")),
                "arrivee_definitive":      _as_bool(ic.get("isArriveeDefinitive")),
            }

            # Course raw (JSON blobs)
            course_rows.append({
                "race_id":                   race_id,
                "file_date":                 file_date,
                "reunion_num":               r_num,
                "course_num":                c_num,
                "ordre_arrivee_json":        _json_or_none(ic.get("ordreArrivee")),
                "performances_detaillees_json": _json_or_none(course_obj.get("performances_detaillees")),
                "masse_enjeu_json":          _json_or_none(course_obj.get("masse_enjeu")),
                "rapports_definitifs_json":  _json_or_none(course_obj.get("rapports_definitifs")),
                "pronostics_json":           _json_or_none(course_obj.get("pronostics")),
                "pronostics_detailles_json": _json_or_none(course_obj.get("pronostics_detailles")),
                "citations_json":            _json_or_none(course_obj.get("citations")),
            })

            # Participants
            for p in pl:
                gp = p.get("gainsParticipant") or {}
                robe = p.get("robe") or {}
                drd = p.get("dernierRapportDirect") or {}
                drr = p.get("dernierRapportReference") or {}
                fin = _finish_position(p.get("ordreArrivee"))

                row = dict(course_common)
                row.update({
                    "num_pmu":                 _as_int(p.get("numPmu")),
                    "nom":                     _as_str(p.get("nom")),
                    "age":                     _as_int(p.get("age")),
                    "sexe":                    _as_str(p.get("sexe")),
                    "race":                    _as_str(p.get("race")),
                    "statut_participant":      _as_str(p.get("statut")),
                    "oeilleres":               _as_str(p.get("oeilleres")),
                    "proprietaire":            _as_str(p.get("proprietaire")),
                    "entraineur":              _as_str(p.get("entraineur")),
                    "driver":                  _as_str(p.get("driver")),
                    "driver_change":           _as_bool(p.get("driverChange")),
                    "indicateur_inedit":       _as_bool(p.get("indicateurInedit")),
                    "nombre_courses":          _as_int(p.get("nombreCourses")),
                    "nombre_victoires":        _as_int(p.get("nombreVictoires")),
                    "nombre_places":           _as_int(p.get("nombrePlaces")),
                    "nombre_places_second":    _as_int(p.get("nombrePlacesSecond")),
                    "nombre_places_troisieme": _as_int(p.get("nombrePlacesTroisieme")),
                    "nom_pere":                _as_str(p.get("nomPere")),
                    "nom_mere":                _as_str(p.get("nomMere")),
                    "nom_pere_mere":           _as_str(p.get("nomPereMere")),
                    "jument_pleine":           _as_bool(p.get("jumentPleine")),
                    "engagement":              _as_bool(p.get("engagement")),
                    "supplement":              _as_int(p.get("supplement")),
                    "poids_condition_monte_change": _as_bool(p.get("poidsConditionMonteChange")),
                    "allure":                  _as_str(p.get("allure")),
                    "musique":                 _as_str(p.get("musique")),
                    "handicap_distance":       _as_int(p.get("handicapDistance")),
                    "handicap_poids":          _as_int(p.get("handicapPoids")),
                    "handicap_valeur":         _as_int(p.get("handicapValeur")),
                    "place_corde":             _as_int(p.get("placeCorde")),
                    "eleveur":                 _as_str(p.get("eleveur")),
                    "temps_obtenu":            _as_int(p.get("tempsObtenu")),
                    "reduction_kilometrique":  _as_int(p.get("reductionKilometrique")),
                    "distance_cheval_precedent": _as_int(p.get("distanceChevalPrecedent")),
                    "pays_entrainement":       _as_str(p.get("paysEntrainement")),
                    "avis_entraineur":         _as_str(p.get("avisEntraineur")),
                    "url_casaque":             _as_str(p.get("urlCasaque")),
                    # derived
                    "finish_position":         fin,
                    "won":                     (fin == 1) if fin is not None else None,
                    "placed":                  (fin is not None and 1 <= fin <= 3),
                    # gains
                    "gains_annee_en_cours":    _as_int(gp.get("gainsAnneeEnCours")),
                    "gains_annee_precedente":  _as_int(gp.get("gainsAnneePrecedente")),
                    "gains_carriere":          _as_int(gp.get("gainsCarriere")),
                    "gains_place":             _as_int(gp.get("gainsPlace")),
                    "gains_victoires":         _as_int(gp.get("gainsVictoires")),
                    # robe
                    "robe_code":               _as_str(robe.get("code")),
                    "robe_libelle_court":      _as_str(robe.get("libelleCourt")),
                    "robe_libelle_long":       _as_str(robe.get("libelleLong")),
                    # drd (live)
                    "drd_rapport":             _as_float(drd.get("rapport")),
                    "drd_favoris":             _as_bool(drd.get("favoris")),
                    "drd_grosse_prise":        _as_bool(drd.get("grossePrise")),
                    "drd_nombre_indicateur_tendance": _as_int(drd.get("nombreIndicateurTendance")),
                    "drd_permutation":         _as_int(drd.get("permutation")),
                    "drd_type_pari":           _as_str(drd.get("typePari")),
                    "drd_type_rapport":        _as_str(drd.get("typeRapport")),
                    "drd_date_rapport_ts":     _as_int(drd.get("dateRapport")),
                    # drr (reference)
                    "drr_rapport":             _as_float(drr.get("rapport")),
                    "drr_favoris":             _as_bool(drr.get("favoris")),
                    "drr_grosse_prise":        _as_bool(drr.get("grossePrise")),
                    "drr_nombre_indicateur_tendance": _as_int(drr.get("nombreIndicateurTendance")),
                    "drr_permutation":         _as_int(drr.get("permutation")),
                    "drr_type_pari":           _as_str(drr.get("typePari")),
                    "drr_type_rapport":        _as_str(drr.get("typeRapport")),
                    "drr_date_rapport_ts":     _as_int(drr.get("dateRapport")),
                })
                part_rows.append(row)

    return part_rows, course_rows, errors


# ── Parquet writer (streaming) ───────────────────────────────────────────────

class StreamWriter:
    def __init__(self, path: Path, schema: pa.Schema, compression: str = "zstd", batch_size: int = 20_000):
        self.path = path
        self.schema = schema
        self._writer = pq.ParquetWriter(str(path), schema, compression=compression)
        self._buffer: list[dict] = []
        self._written = 0
        self._batch_size = batch_size

    def add(self, rows: Iterable[dict]) -> None:
        self._buffer.extend(rows)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        cols = {name: [] for name in self.schema.names}
        for r in self._buffer:
            for name in self.schema.names:
                cols[name].append(r.get(name))
        table = pa.Table.from_pydict(cols, schema=self.schema)
        self._writer.write_table(table)
        self._written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> int:
        self._flush()
        self._writer.close()
        return self._written


# ── Main ─────────────────────────────────────────────────────────────────────

def iter_files(from_date: str | None, to_date: str | None) -> list[Path]:
    all_files = sorted(DATA_DIR.glob("*.json.gz"))
    if from_date:
        all_files = [f for f in all_files if _extract_file_date(f) >= from_date]
    if to_date:
        all_files = [f for f in all_files if _extract_file_date(f) <= to_date]
    return all_files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--to", dest="to_date", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out-participants", default=str(OUT_PARTICIPANTS))
    ap.add_argument("--out-courses", default=str(OUT_COURSES))
    ap.add_argument("--limit", type=int, default=None, help="Debug: only N files")
    args = ap.parse_args()

    files = iter_files(args.from_date, args.to_date)
    if args.limit:
        files = files[: args.limit]

    if not files:
        print("No files found in", DATA_DIR, file=sys.stderr)
        return 1

    print(f"Files          : {len(files)}")
    print(f"Workers        : {args.workers}")
    print(f"Out participants: {args.out_participants}")
    print(f"Out courses    : {args.out_courses}")
    print("-" * 60)

    part_writer = StreamWriter(Path(args.out_participants), PARTICIPANT_SCHEMA, batch_size=20_000)
    course_writer = StreamWriter(Path(args.out_courses), COURSE_SCHEMA, batch_size=500)

    t0 = time.time()
    n_done = 0
    n_err = 0
    total_parts = 0
    total_courses = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_flatten_file, str(p)): p for p in files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                parts, courses, errs = fut.result()
            except Exception:
                traceback.print_exc()
                n_err += 1
                continue
            if errs:
                n_err += errs
            part_writer.add(parts)
            course_writer.add(courses)
            total_parts += len(parts)
            total_courses += len(courses)
            n_done += 1

            if n_done % 50 == 0 or n_done == len(files):
                pct = 100 * n_done / len(files)
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-6)
                eta = (len(files) - n_done) / max(rate, 1e-6)
                print(
                    f"[{n_done:>5}/{len(files)}] {pct:5.1f}%  "
                    f"parts={total_parts:>8}  courses={total_courses:>6}  "
                    f"err={n_err}  rate={rate:5.1f} f/s  eta={eta/60:5.1f} min",
                    flush=True,
                )

    w_parts = part_writer.close()
    w_courses = course_writer.close()

    dt = time.time() - t0
    print("-" * 60)
    print(f"Done in {dt/60:.1f} min")
    print(f"Participants written : {w_parts:,}")
    print(f"Courses written      : {w_courses:,}")
    print(f"Errors               : {n_err}")
    print(f"Participants parquet : {args.out_participants}")
    print(f"Courses parquet      : {args.out_courses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
