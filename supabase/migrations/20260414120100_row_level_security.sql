-- Row Level Security policies.
-- Reference / public data (races, participants, predictions, horses, etc.)
-- is readable by any authenticated user. Only bets_log is user-private.

-- ─────────────────────────────────────────────────────────────────────────────
-- bets_log: private per user
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.bets_log enable row level security;

create policy "bets_log select own"
  on public.bets_log for select
  using (auth.uid() = user_id);

create policy "bets_log insert own"
  on public.bets_log for insert
  with check (auth.uid() = user_id);

create policy "bets_log update own"
  on public.bets_log for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "bets_log delete own"
  on public.bets_log for delete
  using (auth.uid() = user_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Read-only tables for authenticated users (anon keys used by Android app)
-- Service role bypasses RLS anyway; no insert/update policy needed for app.
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.races          enable row level security;
alter table public.participants   enable row level security;
alter table public.predictions    enable row level security;
alter table public.model_versions enable row level security;
alter table public.hippodromes    enable row level security;
alter table public.horses         enable row level security;
alter table public.jockeys        enable row level security;
alter table public.trainers       enable row level security;

create policy "races readable"          on public.races          for select using (true);
create policy "participants readable"   on public.participants   for select using (true);
create policy "predictions readable"    on public.predictions    for select using (true);
create policy "model_versions readable" on public.model_versions for select using (true);
create policy "hippodromes readable"    on public.hippodromes    for select using (true);
create policy "horses readable"         on public.horses         for select using (true);
create policy "jockeys readable"        on public.jockeys        for select using (true);
create policy "trainers readable"       on public.trainers       for select using (true);
