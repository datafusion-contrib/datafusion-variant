SELECT variant_get(data, 'did', 'Utf8View') AS user_id,
       MIN(to_timestamp_micros(variant_get(data, 'time_us', 'Int64'))) AS first_post_ts
FROM bluesky
WHERE variant_get(data, 'kind', 'Utf8View') = 'commit'
  AND variant_get(data, 'commit.operation', 'Utf8View') = 'create'
  AND variant_get(data, 'commit.collection', 'Utf8View') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;
