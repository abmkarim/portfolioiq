select
    ticker,
    trade_date,
    adj_close / lag(adj_close) over (partition by ticker order by trade_date) - 1 as ret
from {{ ref('stg_prices') }}
