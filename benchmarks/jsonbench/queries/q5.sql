-- JSONBench contains positive epoch-microsecond timestamps. Divide each Int64
-- endpoint before subtraction to count millisecond boundaries, as dateDiff does.
SELECT data:did::VARCHAR AS user_id,
       MAX(data:time_us::BIGINT) / 1000
         - MIN(data:time_us::BIGINT) / 1000 AS activity_span
FROM bluesky
WHERE data:kind::VARCHAR = 'commit'
  AND data:commit.operation::VARCHAR = 'create'
  AND data:commit.collection::VARCHAR = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;
