{% macro src(file) -%}
read_parquet('{{ var("data_dir") }}/{{ file }}')
{%- endmacro %}

{% macro report_src(file) -%}
read_parquet('{{ var("reports_dir") }}/{{ file }}')
{%- endmacro %}
