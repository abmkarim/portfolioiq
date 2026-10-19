-- Every panel row must belong to the S&P 500 on its month-end date (no survivorship / look-ahead).
select p.panel_id
from {{ ref('fct_monthly_panel') }} p
where not exists (
    select 1
    from {{ ref('stg_universe') }} u
    where u.ticker = p.ticker
      and p.month_end between u.start_date and u.end_date
)
