-- Candidatos de sentimiento pre-registrados. La seleccion final debe basarse
-- en cobertura y relevancia documental, nunca en el rendimiento predictivo.
WITH base AS (
  SELECT
    DATE(_PARTITIONTIME) AS article_date,
    DocumentIdentifier,
    COALESCE(SourceCommonName, '') AS source_name,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
    UPPER(COALESCE(Themes, '')) AS themes,
    UPPER(COALESCE(V2Organizations, '')) AS organizations,
    `DATE` AS gdelt_timestamp
  FROM `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE _PARTITIONTIME >= TIMESTAMP(@start_date)
    AND _PARTITIONTIME < TIMESTAMP(DATE_ADD(@end_date, INTERVAL 1 DAY))
    AND V2Tone IS NOT NULL
),
deduplicated AS (
  SELECT *
  FROM base
  WHERE DocumentIdentifier IS NOT NULL
    AND tone IS NOT NULL
    AND (
      REGEXP_CONTAINS(organizations, r'(^|;)[^;]*NASDAQ[^;]*,')
    )
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY article_date, DocumentIdentifier
    ORDER BY gdelt_timestamp DESC
  ) = 1
),
labeled AS (
  SELECT
    *,
    REGEXP_CONTAINS(
      organizations,
      r'(^|;)(NASDAQ 100|NASDAQ-100|NASDAQ100|NDX),[0-9]+'
    ) AS is_exact_ndx,
    REGEXP_CONTAINS(themes, r'(^|;)ECON_STOCKMARKET(;|$)') AS is_market_news
  FROM deduplicated
),
expanded AS (
  SELECT 'broad_nasdaq' AS scope, * FROM labeled
  UNION ALL
  SELECT 'nasdaq_market' AS scope, * FROM labeled WHERE is_market_news
  UNION ALL
  SELECT 'exact_ndx' AS scope, * FROM labeled WHERE is_exact_ndx
)
SELECT
  article_date AS Date,
  scope,
  COUNT(*) AS n_articles,
  COUNT(DISTINCT source_name) AS n_sources,
  AVG(tone) AS mean_tone,
  STDDEV_SAMP(tone) AS tone_std,
  APPROX_QUANTILES(tone, 100)[OFFSET(50)] AS median_tone,
  ARRAY_TO_STRING(
    ARRAY_AGG(DocumentIdentifier ORDER BY gdelt_timestamp DESC LIMIT 3),
    ' | '
  ) AS example_urls
FROM expanded
GROUP BY Date, scope
ORDER BY Date, scope
