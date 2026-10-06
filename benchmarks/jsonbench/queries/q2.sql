SELECT variant_get(data, 'commit.collection', 'VARCHAR') AS event,
       COUNT(*) AS count,
       COUNT(DISTINCT variant_get(data, 'did', 'VARCHAR')) AS users
FROM bluesky
WHERE variant_get(data, 'kind', 'VARCHAR') = 'commit'
  AND variant_get(data, 'commit.operation', 'VARCHAR') = 'create'
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;
