-- Trading calendar = dates with at least one observed price.
select distinct trade_date from {{ ref('stg_prices') }}
