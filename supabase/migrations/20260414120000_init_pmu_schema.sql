-- PMU schema: hot data for live prediction + bet tracking.
-- Historical dataset (2.5M rows) stays in Storage / parquet; DB holds only
-- today's races, predictions, user bets, and model metadata.

-- ─────────────────────────────────────────────────────────────────────────────
-- Reference tables (normalization)
-- ─────────────────────────────────────────────────────────────────────────────

create table public.hippodromes (
  id      bigserial primary key,
  name    text unique not null,
  country text default 'FR'
);

create table public.horses (
  id         bigserial primary key,
  horse_key  text unique not null,
  name       text,
  sexe       text,
  first_seen date default current_date
);

create table public.jockeys (
  id   bigserial primary key,
  name text unique not null
);

create table public.trainers (
  id   bigserial primary key,
  name text unique not null
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Races (one row per course)
-- ─────────────────────────────────────────────────────────────────────────────

create table public.races (
  id            bigserial primary key,
  race_id       text unique not null,           -- e.g. "20260414_R1_C3"
  date          date not null,
  reunion       text,
  course        text,
  hippodrome_id bigint references public.hippodromes(id),
  discipline    text,
  distance      int,
  terrain       text,
  nb_partants   int,
  start_time    timestamptz,
  created_at    timestamptz default now()
);
create index races_date_idx          on public.races(date desc);
create index races_hippodrome_idx    on public.races(hippodrome_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Participants (one row per (race, horse))
-- ─────────────────────────────────────────────────────────────────────────────

create table public.participants (
  id                bigserial primary key,
  race_id           bigint not null references public.races(id) on delete cascade,
  num_pmu           int,
  horse_id          bigint references public.horses(id),
  jockey_id         bigint references public.jockeys(id),
  trainer_id        bigint references public.trainers(id),
  age               int,
  musique           text,
  handicap_poids    int,
  deferre           text,
  -- market (pre-race)
  cote_probable     real,
  cote_direct       real,
  tendance          text,
  -- result (filled post-race)
  finish_position   int,
  won               boolean,
  placed            boolean,
  dividende_gagnant numeric,
  dividende_place   numeric,
  -- cached features (optional — full features stay in parquet)
  features_json     jsonb,
  unique(race_id, num_pmu)
);
create index participants_race_idx   on public.participants(race_id);
create index participants_horse_idx  on public.participants(horse_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Model versions (track every retrain)
-- ─────────────────────────────────────────────────────────────────────────────

create table public.model_versions (
  id                 bigserial primary key,
  version            text unique not null,      -- e.g. "2026-04-14"
  trained_at         timestamptz default now(),
  train_start        date,
  train_end          date,
  auc_cv             real,
  n_train_rows       int,
  hyperparams        jsonb,
  feature_importance jsonb,
  storage_path       text,                      -- "models/model_lgbm_2026-04-14.pkl"
  is_active          boolean default false
);

-- Only one active model at a time
create unique index model_versions_active_idx
  on public.model_versions(is_active)
  where is_active = true;

-- ─────────────────────────────────────────────────────────────────────────────
-- Predictions (model output per participant)
-- ─────────────────────────────────────────────────────────────────────────────

create table public.predictions (
  id             bigserial primary key,
  participant_id bigint not null references public.participants(id) on delete cascade,
  model_version  text not null references public.model_versions(version),
  proba_win      real not null,
  edge           real,                          -- proba_win - 1/cote_direct
  strategy_flags jsonb,                         -- {"low_edge": true, "outsider_he": false}
  predicted_at   timestamptz default now(),
  unique(participant_id, model_version)
);
create index predictions_participant_idx on public.predictions(participant_id);
create index predictions_predicted_at_idx on public.predictions(predicted_at desc);

-- ─────────────────────────────────────────────────────────────────────────────
-- Bets log (user-level; RLS below)
-- ─────────────────────────────────────────────────────────────────────────────

create table public.bets_log (
  id             bigserial primary key,
  user_id        uuid references auth.users(id) on delete cascade,
  participant_id bigint references public.participants(id),
  strategy       text,                          -- "LOW_EDGE", "OUTSIDER_HE", "MANUAL"
  stake          numeric not null check (stake > 0),
  cote_taken     real not null,
  placed_at      timestamptz default now(),
  outcome        text default 'pending' check (outcome in ('pending','won','lost','void')),
  pnl            numeric,                       -- filled post-race
  notes          text
);
create index bets_log_user_idx   on public.bets_log(user_id, placed_at desc);
create index bets_log_outcome_idx on public.bets_log(outcome) where outcome = 'pending';
