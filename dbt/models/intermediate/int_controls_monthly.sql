-- Price-based controls at each month end: momentum 12-1, 1-month reversal, 3-month volatility.
with mr as (
    select ticker, month, ret,
        exp(sum(ln(1 + ret)) over (partition by ticker order by month rows between 11 preceding and 1 preceding)) - 1
            as mom_12_1,
        count(ret) over (partition by ticker order by month rows between 11 preceding and 1 preceding) as n_mom
    from {{ ref('int_monthly_returns') }}
),
vol as (
    select ticker, trade_date,
        stddev_samp(ret) over (partition by ticker order by trade_date rows between 62 preceding and current row)
            * sqrt(252) as vol_3m,
        count(ret) over (partition by ticker order by trade_date rows between 62 preceding and current row) as n_vol
    from {{ ref('int_daily_returns') }}
)
select
    m.ticker, m.month,
    m.ret as rev_1m,
    case when m.n_mom = 11 then m.mom_12_1 end as mom_12_1,
    case when v.n_vol >= 40 then v.vol_3m end as vol_3m
from mr m
join {{ ref('int_month_ends') }} me on me.month = m.month
left join vol v on v.ticker = m.ticker and v.trade_date = me.month_end
