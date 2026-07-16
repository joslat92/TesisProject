-- Candidatos de sentimiento pre-registrados. La seleccion final debe basarse
-- en cobertura y relevancia documental, nunca en el rendimiento predictivo.
WITH base AS (
  SELECT
    DATE(_PARTITIONTIME) AS article_date,
    DocumentIdentifier,
    COALESCE(SourceCommonName, '') AS source_name,
    SAFE_CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) AS tone,
    UPPER(COALESCE(V2Organizations, '')) AS organizations,
    UPPER(COALESCE(AllNames, '')) AS all_names,
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
      OR REGEXP_CONTAINS(
        all_names,
        r'(^|;)[^;]*(NASDAQ|NASDAQ 100|NASDAQ-100|INVESCO QQQ|POWERSHARES QQQ)[^;]*,'
      )
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
      organizations || ';' || all_names,
      r'(NASDAQ 100|NASDAQ-100|INVESCO QQQ|POWERSHARES QQQ)'
    ) AS is_strict_ndx
  FROM deduplicated
),
expanded AS (
  SELECT 'broad_nasdaq' AS scope, * FROM labeled
  UNION ALL
  SELECT 'strict_ndx' AS scope, * FROM labeled WHERE is_strict_ndx
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
