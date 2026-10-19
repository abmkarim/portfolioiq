select
    cast(date as date) as trade_date,
    ticker,
    cast(adj_close as double) as adj_close,
    cast(close_raw as double) as close_raw,
    cast(volume as double) as volume
from {{ src('prices.parquet') }}
where adj_close > 0
