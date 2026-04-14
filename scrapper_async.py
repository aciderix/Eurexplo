#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMU API Scraper Ultra-Fast -- asyncio + aiohttp
Sortie : 1 fichier .json.gz par jour, structure par reunion (info_reunion deduplique).
Usage : python scrapper_async.py [DDMMYYYY DDMMYYYY]
        python scrapper_async.py 01012014 14042026
"""
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import aiohttp
import gzip
import json
import os
import time
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = "https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
OUTPUT_DIR = "pmu_data"
HEADERS    = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

MAX_CONCURRENT_REQUESTS = 60
MAX_CONCURRENT_DAYS     = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HTTP ──────────────────────────────────────────────────────────────────────

async def fetch_json(session, url, sem):
    async with sem:
        for attempt in range(4):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=20), ssl=False) as r:
                    if r.status == 200:
                        return await r.json(content_type=None)
                    if r.status == 404:
                        return None
                    if r.status == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < 3:
                    await asyncio.sleep(1 + attempt)
    return None


# ── Scraping d'une course ─────────────────────────────────────────────────────

async def fetch_race(session, sem, date_str, r_num, c_num):
    """Recupere tous les endpoints d'une course en parallele (sauf info_reunion, gere au niveau reunion)."""
    course_url = f"{BASE}/{date_str}/R{r_num}/C{c_num}"

    urls = [
        course_url,
        f"{course_url}/participants",
        f"{course_url}/performances-detaillees/pretty",
        f"{course_url}/masse-enjeu",
        f"{course_url}/rapports-definitifs",
        f"{course_url}/citations",
        f"{course_url}/pronostics",
        f"{course_url}/pronostics-detailles",
    ]

    data = await asyncio.gather(*[fetch_json(session, u, sem) for u in urls])

    if data[0] is None:
        return None

    return {
        "course":                 c_num,
        "info_course":            data[0],
        "participants":           data[1],
        "performances_detaillees": data[2],
        "masse_enjeu":            data[3],
        "rapports_definitifs":    data[4],
        "citations":              data[5],
        "pronostics":             data[6],
        "pronostics_detailles":   data[7],
    }


# ── Scraping d'une journee ────────────────────────────────────────────────────

def _filepath(date_str):
    d_obj = datetime.strptime(date_str, "%d%m%Y")
    base  = d_obj.strftime("%Y-%m-%d")
    return (
        os.path.join(OUTPUT_DIR, base + ".json.gz"),
        os.path.join(OUTPUT_DIR, base + ".json"),   # ancien format (compat reprise)
    )

def _save(filepath, data):
    raw = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    with gzip.open(filepath, 'wb', compresslevel=6) as f:
        f.write(raw)


async def process_day(session, sem, date_str, day_sem, progress):
    gz_path, json_path = _filepath(date_str)

    # Reprise auto : .json.gz ou .json deja present
    if os.path.exists(gz_path) or os.path.exists(json_path):
        progress["skipped"] += 1
        return

    async with day_sem:
        # 1. Programme du jour
        prog = await fetch_json(session, f"{BASE}/{date_str}", sem)
        if not prog:
            _save(gz_path, [])
            progress["empty"] += 1
            progress["done"]  += 1
            _print_progress(progress, date_str, 0, "vide")
            return

        programme = prog.get("programme", prog)

        # 2. Pour chaque reunion, on fetch info_reunion UNE FOIS + toutes ses courses en parallele
        reunion_tasks = []
        for reunion in programme.get("reunions", []):
            r_num   = reunion.get("numOfficiel", 0)
            c_nums  = [c.get("numOrdre", c.get("numExterne", 0)) for c in reunion.get("courses", [])]
            reunion_tasks.append((r_num, c_nums))

        if not reunion_tasks:
            _save(gz_path, [])
            progress["empty"] += 1
            progress["done"]  += 1
            _print_progress(progress, date_str, 0, "vide")
            return

        # Lance info_reunion + toutes les courses de toutes les reunions en parallele
        all_coros = []
        # (reunion_index, type, r_num, c_num_or_None)
        meta = []

        for r_num, c_nums in reunion_tasks:
            all_coros.append(fetch_json(session, f"{BASE}/{date_str}/R{r_num}", sem))
            meta.append(("reunion", r_num, None))
            for c_num in c_nums:
                all_coros.append(fetch_race(session, sem, date_str, r_num, c_num))
                meta.append(("course", r_num, c_num))

        results = await asyncio.gather(*all_coros)

        # Reconstruction structure par reunion
        reunions_map = {}
        for (kind, r_num, c_num), result in zip(meta, results):
            if kind == "reunion":
                reunions_map.setdefault(r_num, {"reunion": r_num, "info_reunion": result, "courses": []})
            else:
                if result is not None:
                    reunions_map.setdefault(r_num, {"reunion": r_num, "info_reunion": None, "courses": []})
                    reunions_map[r_num]["courses"].append(result)

        day_data      = [v for v in reunions_map.values() if v["courses"]]
        total_courses = sum(len(r["courses"]) for r in day_data)

        _save(gz_path, day_data)

        progress["scraped"]       += 1
        progress["total_courses"] += total_courses
        progress["done"]          += 1
        _print_progress(progress, date_str, total_courses, "ok")


def _print_progress(p, date_str, nb, status):
    d_fmt  = datetime.strptime(date_str, "%d%m%Y").strftime("%Y-%m-%d")
    pct    = 100 * p["done"] / p["total"] if p["total"] else 0
    marker = "+" if status == "ok" else ("~" if status == "vide" else "!")
    print(
        f"[{p['done']:>5}/{p['total']}] {pct:5.1f}%  {marker} {d_fmt}  "
        f"{nb:>3} courses   |  total: {p['total_courses']}",
        flush=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def date_range(start_str, end_str):
    d   = datetime.strptime(start_str, "%d%m%Y")
    end = datetime.strptime(end_str,   "%d%m%Y")
    while d <= end:
        yield d.strftime("%d%m%Y")
        d += timedelta(days=1)


async def run(start_date, end_date):
    dates = list(date_range(start_date, end_date))

    print("=" * 60)
    print(f"  PMU Scraper — asyncio | sortie: .json.gz")
    print(f"  Periode  : {start_date} -> {end_date}")
    print(f"  Journees : {len(dates)}")
    print(f"  Sortie   : ./{OUTPUT_DIR}/")
    print("=" * 60)

    progress = {"total": len(dates), "done": 0, "skipped": 0,
                "scraped": 0, "empty": 0, "total_courses": 0}

    sem     = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    day_sem = asyncio.Semaphore(MAX_CONCURRENT_DAYS)

    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS, ttl_dns_cache=300, enable_cleanup_closed=True
    )

    t0 = time.time()
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:
        try:
            async with session.get("https://www.pmu.fr/", ssl=False,
                                   timeout=aiohttp.ClientTimeout(total=15)) as _:
                pass
            print("  Warmup pmu.fr OK\n")
        except Exception:
            print("  Warmup pmu.fr echoue (on continue)\n")

        await asyncio.gather(*[
            process_day(session, sem, d, day_sem, progress) for d in dates
        ])

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Termine en {elapsed/60:.1f} min")
    print(f"  Journees : {progress['scraped']} ok | {progress['empty']} vides | {progress['skipped']} skipped")
    print(f"  Courses  : {progress['total_courses']}")
    print("=" * 60)


def main():
    args = sys.argv[1:]
    if len(args) >= 2:
        start_date, end_date = args[0], args[1]
    else:
        end_date   = datetime.now().strftime("%d%m%Y")
        start_date = "01012014"
    asyncio.run(run(start_date, end_date))


if __name__ == "__main__":
    main()

