"""Fetch all Euromillions draws from the API and save as numpy arrays."""
import json
import urllib.request
import numpy as np
import time

API_BASE = "https://euromillions.api.pedromealha.dev"

def fetch_draws_by_year(year, max_retries=4):
    url = f"{API_BASE}/v1/draws?year={year}"
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  Retry {attempt+1}/{max_retries} for {year} (wait {wait}s): {e}")
            time.sleep(wait)
    print(f"  FAILED {year} after {max_retries} retries")
    return []

def fetch_all_draws():
    all_draws = []
    for year in range(2004, 2027):
        draws = fetch_draws_by_year(year)
        print(f"  {year}: {len(draws)} draws")
        all_draws.extend(draws)
        time.sleep(2)

    # Sort by date
    all_draws.sort(key=lambda d: d['date'])
    return all_draws

def to_numpy(draws):
    dates = []
    numbers = np.zeros((len(draws), 5), dtype=np.int8)
    stars = np.zeros((len(draws), 2), dtype=np.int8)
    has_winner = np.zeros(len(draws), dtype=bool)
    jackpots = np.zeros(len(draws), dtype=np.float64)

    for i, d in enumerate(draws):
        dates.append(d['date'])
        nums = sorted([int(n) for n in d['numbers']])
        st = sorted([int(s) for s in d['stars']])
        numbers[i] = nums
        stars[i] = st
        has_winner[i] = d.get('has_winner', False)
        # Extract jackpot amount (5+2 prize)
        for p in d.get('prizes', []):
            if p.get('matched_numbers') == 5 and p.get('matched_stars') == 2:
                jackpots[i] = p.get('prize', 0)

    return dates, numbers, stars, has_winner, jackpots

if __name__ == '__main__':
    print("Fetching all draws...")
    draws = fetch_all_draws()
    print(f"\nTotal: {len(draws)} draws")

    dates, numbers, stars, has_winner, jackpots = to_numpy(draws)

    np.savez('/home/user/Eurexplo/euromillions_data.npz',
             numbers=numbers, stars=stars, has_winner=has_winner,
             jackpots=jackpots)

    # Save dates separately (strings)
    with open('/home/user/Eurexplo/dates.json', 'w') as f:
        json.dump(dates, f)

    print(f"Saved: numbers {numbers.shape}, stars {stars.shape}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(f"Winners: {has_winner.sum()}/{len(has_winner)}")
