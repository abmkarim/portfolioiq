-- Rebalancing dates: last trading day of each calendar month.
select date_trunc('month', trade_date) as month, max(trade_date) as month_end
from {{ ref('int_trading_days') }}
group by 1
