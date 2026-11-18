SELECT data:did::VARCHAR AS user_id,
       MIN(to_timestamp_micros(data:time_us::BIGINT)) AS first_post_ts
FROM bluesky
WHERE data:kind::VARCHAR = 'commit'
  AND data:commit.operation::VARCHAR = 'create'
  AND data:commit.collection::VARCHAR = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;
