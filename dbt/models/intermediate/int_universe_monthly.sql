-- Point-in-time membership: (month_end, ticker) for tickers in the index ON the month-end date.
select m.month, m.month_end, u.ticker, u.cik, u.map_source
from {{ ref('int_month_ends') }} m
join {{ ref('stg_universe') }} u
  on m.month_end between u.start_date and u.end_date
