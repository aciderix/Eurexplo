-- Storage buckets for model artifacts and parquet datasets.
-- Buckets are private by default; access via service_role only.

insert into storage.buckets (id, name, public)
values
  ('models',     'models',     false),
  ('datasets',   'datasets',   false),
  ('historical', 'historical', false)
on conflict (id) do nothing;

-- RLS on storage.objects is enabled by default.
-- Service role bypasses RLS; no policy needed for backend-only access.
-- If we later want the Android app to download models directly (instead of
-- going through an edge function), add a select policy here.
