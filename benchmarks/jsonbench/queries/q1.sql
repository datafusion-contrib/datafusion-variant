SELECT variant_get(data, 'commit.collection', 'VARCHAR') AS event,
       COUNT(*) AS count
FROM bluesky
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;
