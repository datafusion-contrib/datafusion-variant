SELECT variant_get(data, 'commit.collection', 'Utf8View') AS event,
       COUNT(*) AS count
FROM bluesky
GROUP BY event
ORDER BY count DESC, event ASC NULLS FIRST;
