SELECT data:commit.collection::VARCHAR AS event,
       EXTRACT(HOUR FROM to_timestamp_micros(data:time_us::BIGINT)) AS hour_of_day,
       COUNT(*) AS count
FROM bluesky
WHERE data:kind::VARCHAR = 'commit'
  AND data:commit.operation::VARCHAR = 'create'
  AND data:commit.collection::VARCHAR IN (
      'app.bsky.feed.post', 'app.bsky.feed.repost', 'app.bsky.feed.like'
  )
GROUP BY event, hour_of_day
ORDER BY hour_of_day ASC NULLS FIRST, event;
