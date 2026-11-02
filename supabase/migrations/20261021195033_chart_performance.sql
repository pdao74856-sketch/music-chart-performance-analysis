-- 01_chart_performance.sql
-- Owner: Member 1 (Report 3)
-- Grain: 1 row = 1 (spotify_track_id, market). Re-runnable (idempotent upsert).
--
-- Official definitions
--   lifetime_average_rank = AVG(rank) over ALL valid chart_weekly rows of the pair,
--                           within the 2021-01-01 .. 2026 study window (crawl scope).
--   chart_score           = 101 - lifetime_average_rank
--                           NOTE: Kworb chart is Top 200, so chart_score is negative
--                           when lifetime_average_rank > 101.
--   peak_rank             = MIN(rank)
--   weeks_on_chart        = COUNT(*) of valid weekly rows
--   weeks_in_top10        = COUNT(*) FILTER (rank BETWEEN 1 AND 10)
-- Pairs with no chart_weekly rows get NO row (never ChartScore = 0).
-- Not used by song_quarter_sampling. Do NOT use these columns as regression predictors.

create table if not exists public.chart_performance (
    spotify_track_id      text        not null references public.songs(spotify_track_id) on delete cascade,
    market                text        not null check (market in ('VN', 'US', 'KR')),
    lifetime_average_rank numeric     not null check (lifetime_average_rank between 1 and 200),
    chart_score           numeric     not null,
    peak_rank             integer     not null check (peak_rank between 1 and 200),
    weeks_on_chart        integer     not null check (weeks_on_chart >= 1),
    weeks_in_top10        integer     not null check (weeks_in_top10 >= 0),
    created_at            timestamptz not null default now(),
    primary key (spotify_track_id, market),
    constraint chk_chart_score_formula
        check (abs(chart_score - (101 - lifetime_average_rank)) < 1e-9),
    constraint chk_top10_le_weeks check (weeks_in_top10 <= weeks_on_chart)
);

create index if not exists idx_chart_performance_market on public.chart_performance (market);

-- Populate / refresh (safe to re-run: ON CONFLICT updates, no duplicates)
insert into public.chart_performance
    (spotify_track_id, market, lifetime_average_rank, chart_score,
     peak_rank, weeks_on_chart, weeks_in_top10, created_at)
select
    cw.spotify_track_id,
    cw.market,
    avg(cw.rank)::numeric                          as lifetime_average_rank,
    101 - avg(cw.rank)::numeric                    as chart_score,
    min(cw.rank)                                   as peak_rank,
    count(*)                                       as weeks_on_chart,
    count(*) filter (where cw.rank between 1 and 10) as weeks_in_top10,
    now()
from public.chart_weekly cw
join public.songs s on s.spotify_track_id = cw.spotify_track_id
where cw.market in ('VN', 'US', 'KR')
  and cw.rank between 1 and 200
  and cw.chart_date is not null
group by cw.spotify_track_id, cw.market
on conflict (spotify_track_id, market) do update set
    lifetime_average_rank = excluded.lifetime_average_rank,
    chart_score           = excluded.chart_score,
    peak_rank             = excluded.peak_rank,
    weeks_on_chart        = excluded.weeks_on_chart,
    weeks_in_top10        = excluded.weeks_in_top10,
    created_at            = now();
