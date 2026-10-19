select
    cast(cik as bigint) as cik,
    cast(filed_date as date) as filed_date,
    cast(shares as double) as shares_outstanding,
    source
from {{ src('shares.parquet') }}
where shares > 0
