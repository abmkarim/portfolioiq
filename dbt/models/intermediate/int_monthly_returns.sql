-- Monthly returns from month-end adjusted closes, with the documented delisting rule:
-- if a series stops more than 7 days before the end of the data, the delisting return
-- (0 % for acquisitions, -30 % otherwise; configurable) is compounded into the final month.
with last_obs as (
    select ticker, date_trunc('month', trade_date) as month,
           max(trade_date) as last_date,
           arg_max(adj_close, trade_date) as adj_close
    from {{ ref('stg_prices') }}
    group by 1, 2
),
series_end as (
    select ticker, max(trade_date) as series_last_date from {{ ref('stg_prices') }} group by 1
),
global_end as (select max(trade_date) as data_end from {{ ref('stg_prices') }}),
causes as (
    select ticker, any_value(removal_cause) as removal_cause from {{ ref('stg_universe') }} group by 1
),
rets as (
    select
        l.ticker, l.month, l.last_date,
        case when date_diff('month', lag(l.month) over w, l.month) = 1
             then l.adj_close / lag(l.adj_close) over w - 1 end as ret_raw
    from last_obs l
    window w as (partition by l.ticker order by l.month)
)
select
    r.ticker, r.month, r.last_date, r.ret_raw,
    (s.series_last_date < g.data_end - interval 7 day and r.last_date = s.series_last_date) as is_delisting_month,
    case
        when s.series_last_date < g.data_end - interval 7 day and r.last_date = s.series_last_date
        then (1 + coalesce(r.ret_raw, 0)) * (1 + case when c.removal_cause = 'acquisition'
                 then {{ var('delist_return_acquisition') }} else {{ var('delist_return_unknown') }} end) - 1
        else r.ret_raw
    end as ret
from rets r
join series_end s using (ticker)
cross join global_end g
left join causes c using (ticker)
