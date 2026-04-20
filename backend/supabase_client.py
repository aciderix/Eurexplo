"""Supabase client singleton for PMU backend scripts.

Reads credentials from environment (or .env file). The service-role key
bypasses RLS and is only safe to use in server-side code — NEVER in the
Android app (which uses the anon / publishable key).
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Return a cached service-role Supabase client."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set. "
            "Copy .env.example to .env and fill in values."
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


if __name__ == "__main__":
    # Smoke test: list a few races
    c = get_client()
    res = c.table("races").select("race_id, date").limit(5).execute()
    print(f"Races found: {len(res.data)}")
    for r in res.data:
        print(f"  {r['date']}  {r['race_id']}")
