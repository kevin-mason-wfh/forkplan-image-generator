-- Apply explicitly, never from the application (after 002). Read-only view: tokens and estimated cost per attempt.
-- Tokens are authoritative provider counts; cost_usd values are estimates from pricing.py (null when no rate is known).
BEGIN;
CREATE OR REPLACE VIEW catalog_image_generation_costs AS
SELECT
    id AS attempt_id, resource_id, attempt_number, stage,
    text_usage->>'model'  AS text_model,
    COALESCE((text_usage->>'input_tokens')::bigint, 0)        AS text_input_tokens,
    COALESCE((text_usage->>'output_text_tokens')::bigint, 0)  AS text_output_tokens,
    COALESCE((text_usage->>'reasoning_tokens')::bigint, 0)    AS text_reasoning_tokens,
    image_usage->>'model' AS image_model,
    COALESCE((image_usage->>'input_tokens')::bigint, 0)         AS image_input_tokens,
    COALESCE((image_usage->>'output_image_tokens')::bigint, 0)  AS image_output_tokens,
    COALESCE((text_usage->>'input_tokens')::bigint, 0) + COALESCE((text_usage->>'output_text_tokens')::bigint, 0)
      + COALESCE((image_usage->>'input_tokens')::bigint, 0) + COALESCE((image_usage->>'output_text_tokens')::bigint, 0)
      + COALESCE((image_usage->>'output_image_tokens')::bigint, 0)  AS total_tokens,
    (text_usage->>'cost_usd')::numeric  AS text_cost_usd,
    (image_usage->>'cost_usd')::numeric AS image_cost_usd,
    COALESCE((text_usage->>'cost_usd')::numeric, 0) + COALESCE((image_usage->>'cost_usd')::numeric, 0) AS total_cost_usd
FROM catalog_image_generations;
-- Example: spend per day
--   SELECT date_trunc('day', completed_at) AS day, count(*), sum(total_tokens), sum(total_cost_usd)
--   FROM catalog_image_generations g JOIN catalog_image_generation_costs c ON c.attempt_id = g.id GROUP BY 1;
COMMIT;
