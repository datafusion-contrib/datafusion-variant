SELECT data:commit.collection::VARCHAR AS event,
       COUNT(*) AS count,
       COUNT(DISTINCT data:did::VARCHAR) AS users
FROM bluesky
WHERE data:kind::VARCHAR = 'commit'
  AND data:commit.operation::VARCHAR = 'create'
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;
