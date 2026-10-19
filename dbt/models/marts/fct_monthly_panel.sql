-- Monthly modelling panel: one row per (month_end, ticker) for index members AT that date.
-- The latest 10-K attached to a row must be available (feature_available_date <= month_end).
with base as (
    select um.month, um.month_end, um.ticker, um.cik
    from {{ ref('int_universe_monthly') }} um
),
filings as (
    select cik, accession, acceptance_datetime, feature_available_date
    from {{ ref('int_filings_available') }}
    where feature_available_date is not null
),
with_filing as (
    select b.*, f.accession, f.acceptance_datetime, f.feature_available_date
    from base b
    asof left join filings f
      on b.cik = f.cik and b.month_end >= f.feature_available_date
)
select
    w.ticker || '_' || strftime(w.month_end, '%Y-%m-%d') as panel_id,
    w.month,
    w.month_end,
    w.ticker,
    w.cik,
    coalesce(d.ff12, 12) as ff12,
    coalesce(d.ff12_name, 'Other') as ff12_name,
    r0.ret as ret_1m,
    r1.ret as ret_fwd_1m,
    coalesce(r1.is_delisting_month, false) as fwd_is_delisting,
    c.mom_12_1,
    c.vol_3m,
    c.rev_1m,
    ln(nullif(s.market_cap, 0)) as log_mcap,
    case when w.acceptance_datetime >= w.month_end - interval {{ var('max_filing_age_months') }} month
         then w.accession end as accession,
    case when w.acceptance_datetime >= w.month_end - interval {{ var('max_filing_age_months') }} month
         then w.acceptance_datetime end as filing_acceptance_datetime,
    case when w.acceptance_datetime >= w.month_end - interval {{ var('max_filing_age_months') }} month
         then w.feature_available_date end as feature_available_date,
    (w.cik is not null and r0.ret is not null) as is_investable
from with_filing w
left join {{ ref('dim_company') }} d on d.cik = w.cik
left join {{ ref('int_monthly_returns') }} r0 on r0.ticker = w.ticker and r0.month = w.month
left join {{ ref('int_monthly_returns') }} r1 on r1.ticker = w.ticker and r1.month = w.month + interval 1 month
left join {{ ref('int_controls_monthly') }} c on c.ticker = w.ticker and c.month = w.month
left join {{ ref('int_size_monthly') }} s on s.ticker = w.ticker and s.month = w.month
