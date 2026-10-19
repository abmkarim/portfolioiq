select
    cast(cik as bigint) as cik,
    accession,
    form,
    cast(filing_date as date) as filing_date,
    cast(report_date as date) as report_date,
    cast(acceptance_datetime as timestamp) as acceptance_datetime,
    coalesce(item1a_ok, false) as item1a_ok,
    coalesce(item7_ok, false) as item7_ok,
    item1a_chars,
    item7_chars
from {{ src('filings_index.parquet') }}
where downloaded
