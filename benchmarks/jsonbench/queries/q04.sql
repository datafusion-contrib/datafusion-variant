SELECT variant_get(data, 'did', 'VARCHAR') AS user_id,
       MIN(to_timestamp_micros(variant_get(data, 'time_us', 'BIGINT'))) AS first_post_ts
FROM bluesky
WHERE variant_get(data, 'kind', 'VARCHAR') = 'commit'
  AND variant_get(data, 'commit.operation', 'VARCHAR') = 'create'
  AND variant_get(data, 'commit.collection', 'VARCHAR') = 'app.bsky.feed.post'
GROUP BY user_id
ORDER BY first_post_ts ASC NULLS LAST, user_id ASC NULLS FIRST
LIMIT 3;
