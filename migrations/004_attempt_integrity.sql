-- Apply explicitly, never from the application (after 001-003). REQUIRED by the current code (it writes description_version).
BEGIN;

-- Which description instructions produced the text (the image prompt version is already in prompt_version).
ALTER TABLE catalog_image_generations ADD COLUMN IF NOT EXISTS description_version text;

-- Lease columns are always set or cleared together (claim, insert and release/retire all do this).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'catalog_image_generations_lease_consistent'
                     AND conrelid = 'catalog_image_generations'::regclass) THEN
        ALTER TABLE catalog_image_generations ADD CONSTRAINT catalog_image_generations_lease_consistent
            CHECK ((lease_owner IS NULL) = (lease_token IS NULL) AND (lease_token IS NULL) = (lease_expires_at IS NULL));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conname = 'catalog_image_generations_usage_is_object'
                     AND conrelid = 'catalog_image_generations'::regclass) THEN
        ALTER TABLE catalog_image_generations ADD CONSTRAINT catalog_image_generations_usage_is_object
            CHECK ((text_usage IS NULL OR jsonb_typeof(text_usage) = 'object')
               AND (image_usage IS NULL OR jsonb_typeof(image_usage) = 'object'));
    END IF;
END $$;

-- The candidate scan filters by stage and orders by age; keep it off the full table as history grows.
CREATE INDEX IF NOT EXISTS catalog_image_generations_open_by_age
    ON catalog_image_generations (created_at, id)
    WHERE stage IN ('selected', 'described', 'prompt_built', 'generating', 'generated',
                    'processed', 'presigned', 'uploaded', 'failed');

-- Cost view: same columns as before, plus description_version appended (CREATE OR REPLACE may only add at the end).
CREATE OR REPLACE VIEW catalog_image_generation_costs AS
SELECT
    id AS attempt_id, resource_id, attempt_number, stage, prompt_version, created_at, completed_at,
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
    COALESCE((text_usage->>'cost_usd')::numeric, 0) + COALESCE((image_usage->>'cost_usd')::numeric, 0) AS total_cost_usd,
    description_version
FROM catalog_image_generations;
COMMIT;
