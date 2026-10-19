{{ config(tags=['post_model']) }}
-- Monthly net returns of the strategy and benchmarks (built after the backtest step).
select
    cast(month_end as date) as month_end,
    strategy,
    cast(ret_net as double) as ret_net,
    cast(ret_gross as double) as ret_gross,
    cast(turnover as double) as turnover,
    cast(cost as double) as cost,
    cast(gross_exposure as double) as gross_exposure,
    strategy || '_' || strftime(cast(month_end as date), '%Y-%m') as row_id
from {{ report_src('backtest_monthly_' ~ var('run_name') ~ '.parquet') }}
