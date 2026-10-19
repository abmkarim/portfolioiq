select
    ticker,
    cast(start_date as date) as start_date,
    cast(end_date as date) as end_date,
    cast(cik as bigint) as cik,
    map_source,
    name as company_name_hint,
    removal_cause
from {{ src('universe.parquet') }}
