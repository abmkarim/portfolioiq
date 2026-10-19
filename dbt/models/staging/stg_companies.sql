select
    cast(cik as bigint) as cik,
    company_name,
    sic,
    sic_description,
    cast(ff12 as integer) as ff12,
    ff12_name
from {{ src('sectors.parquet') }}
