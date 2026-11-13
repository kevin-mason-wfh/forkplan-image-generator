-- Apply explicitly, never from the application. Records provider token usage and priced cost per call.
BEGIN;
ALTER TABLE catalog_image_generations
    ADD COLUMN IF NOT EXISTS text_usage jsonb,
    ADD COLUMN IF NOT EXISTS image_usage jsonb;
COMMENT ON COLUMN catalog_image_generations.text_usage IS
    'Description call: {provider, model, input_tokens, output_text_tokens, output_image_tokens, cost_usd|null}';
COMMENT ON COLUMN catalog_image_generations.image_usage IS
    'Image call, same shape. cost_usd is an estimate from pricing.py rates; tokens are authoritative.';
-- Example: total estimated spend per attempt
--   SELECT id, resource_id, COALESCE((text_usage->>'cost_usd')::numeric,0)
--        + COALESCE((image_usage->>'cost_usd')::numeric,0) AS cost_usd
--   FROM catalog_image_generations;
COMMIT;
