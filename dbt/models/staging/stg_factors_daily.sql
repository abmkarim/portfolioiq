select cast(date as date) as trade_date, mkt_rf, smb, hml, rmw, cma, mom, rf
from read_parquet('{{ var("interim_dir") }}/factors_daily.parquet')
