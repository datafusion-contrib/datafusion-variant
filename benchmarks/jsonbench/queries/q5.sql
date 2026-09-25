-- JSONBench contains positive epoch-microsecond timestamps. Divide each Int64
-- endpoint before subtraction to count millisecond boundaries, as dateDiff does.
SELECT variant_get(data, 'did', 'Utf8View') AS user_id,
       MAX(variant_get(data, 'time_us', 'Int64')) / 1000
         - MIN(variant_get(data, 'time_us', 'Int64')) / 1000 AS activity_span
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8View') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY activity_span DESC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;
