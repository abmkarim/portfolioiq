-- One row per company (CIK) of the historical universe.
with u as (
    select cik,
           string_agg(distinct ticker, ',' order by ticker) as tickers,
           min(start_date) as first_member_date,
           max(end_date) as last_member_date,
           any_value(map_source) as map_source
    from {{ ref('stg_universe') }}
    where cik is not null
    group by cik
)
select
    u.cik,
    coalesce(c.company_name, 'unknown') as company_name,
    c.sic,
    c.sic_description,
    coalesce(c.ff12, 12) as ff12,
    coalesce(c.ff12_name, 'Other') as ff12_name,
    u.tickers,
    u.first_member_date,
    u.last_member_date,
    u.map_source
from u
left join {{ ref('stg_companies') }} c using (cik)
