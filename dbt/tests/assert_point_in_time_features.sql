-- Point-in-time guard: every filing attached to a panel row must have been accepted strictly
-- before the date its features become available, and be available on or before the month end.
select panel_id, month_end, accession, filing_acceptance_datetime, feature_available_date
from {{ ref('fct_monthly_panel') }}
where accession is not null
  and not (
        cast(feature_available_date as timestamp) > filing_acceptance_datetime
    and feature_available_date <= month_end
  )
