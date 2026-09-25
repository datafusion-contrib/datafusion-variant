SELECT variant_get(data, 'commit.collection', 'Utf8View') AS event,
       EXTRACT(HOUR FROM to_timestamp_micros(variant_get(data, 'time_us', 'Int64'))) AS hour_of_day,
       COUNT(*) AS count
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8View') IN (
      'app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like'
  )
GROUP BY event, hour_of_day
ORDER BY hour_of_day ASC NULLS FIRST, event;
