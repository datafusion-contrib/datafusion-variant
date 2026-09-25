SELECT variant_get(data, 'commit.collection', 'Utf8View') AS event,
       COUNT(*) AS count,
       COUNT(DISTINCT variant_get(data, 'did', 'Utf8View')) AS users
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;
