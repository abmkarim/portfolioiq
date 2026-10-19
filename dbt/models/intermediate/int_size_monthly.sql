-- Market cap ~ unadjusted close x latest shares outstanding filed strictly before the month end
-- (point in time; values older than 18 months are ignored).
with px as (
    select p.ticker, me.month, me.month_end, p.close_raw
    from {{ ref('int_month_ends') }} me
    join {{ ref('stg_prices') }} p on p.trade_date = me.month_end
),
u as (select distinct ticker, cik from {{ ref('stg_universe') }} where cik is not null),
sh as (
    select px.ticker, px.month, px.month_end, px.close_raw,
           arg_max(s.shares_outstanding, s.filed_date) as shares
    from px
    join u using (ticker)
    join {{ ref('stg_shares') }} s
      on s.cik = u.cik and s.filed_date < px.month_end and s.filed_date >= px.month_end - interval 18 month
    group by 1, 2, 3, 4
)
select ticker, month, shares * close_raw as market_cap
from sh
