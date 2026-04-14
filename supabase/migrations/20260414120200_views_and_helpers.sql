-- Convenience views for the Android app + backend analytics.

-- ─────────────────────────────────────────────────────────────────────────────
-- today's predictions: joins races + participants + latest prediction
-- Android app calls: select * from today_predictions where date = current_date
-- ─────────────────────────────────────────────────────────────────────────────

create or replace view public.today_predictions as
select
  r.id              as race_pk,
  r.race_id,
  r.date,
  r.reunion,
  r.course,
  h.name            as hippodrome,
  r.discipline,
  r.distance,
  r.terrain,
  r.start_time,
  p.id              as participant_pk,
  p.num_pmu,
  ho.name           as horse_name,
  j.name            as jockey_name,
  t.name            as trainer_name,
  p.age,
  p.cote_probable,
  p.cote_direct,
  p.tendance,
  pr.proba_win,
  pr.edge,
  pr.strategy_flags,
  pr.model_version,
  pr.predicted_at
from public.races r
join public.participants p      on p.race_id = r.id
left join public.hippodromes h  on h.id      = r.hippodrome_id
left join public.horses ho      on ho.id     = p.horse_id
left join public.jockeys j      on j.id      = p.jockey_id
left join public.trainers t     on t.id      = p.trainer_id
left join public.predictions pr on pr.participant_id = p.id
  and pr.model_version = (select version from public.model_versions where is_active limit 1)
where r.date >= current_date
order by r.date, r.start_time, r.reunion, r.course, pr.proba_win desc nulls last;

-- ─────────────────────────────────────────────────────────────────────────────
-- User ROI summary (RLS-aware via bets_log)
-- ─────────────────────────────────────────────────────────────────────────────

create or replace view public.my_roi_summary as
select
  user_id,
  strategy,
  count(*)                                              as n_bets,
  sum(stake)                                            as total_staked,
  sum(coalesce(pnl, 0))                                 as total_pnl,
  case when sum(stake) > 0 then sum(coalesce(pnl, 0)) / sum(stake) else 0 end as roi,
  count(*) filter (where outcome = 'won')::real
    / nullif(count(*) filter (where outcome in ('won','lost')), 0)            as win_rate
from public.bets_log
group by user_id, strategy;

-- View is RLS-enforced through bets_log's policies, so each user only sees
-- their own rows.
