{{ config(tags=['post_model']) }}
-- Mean |SHAP| per feature and walk-forward fold (built after training).
select
    cast(fold_year as integer) as fold_year,
    feature,
    feature_group,
    cast(mean_abs_shap as double) as mean_abs_shap,
    cast(rank as integer) as importance_rank,
    feature || '_' || cast(fold_year as varchar) as row_id
from {{ report_src('shap_importance_' ~ var('run_name') ~ '.parquet') }}
