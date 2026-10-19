-- Anti look-ahead: a 10-K is usable from the first trading day STRICTLY after its acceptance date.
select
    f.*,
    (select min(t.trade_date) from {{ ref('int_trading_days') }} t
      where t.trade_date > cast(f.acceptance_datetime as date)) as feature_available_date
from {{ ref('stg_filings') }} f
