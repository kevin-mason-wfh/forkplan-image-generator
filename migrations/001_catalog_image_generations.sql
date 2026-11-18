-- Image generation tracking for the forkplan-images CLI. Apply once, explicitly (never from the application), with a role
-- that can create tables and views. Catalog tables are only read by the application; this is the only table it writes.
-- Rollback: DROP VIEW catalog_image_generation_costs; DROP TABLE catalog_image_generations;
--           DROP FUNCTION catalog_image_usage_is_valid(jsonb);
--
-- One row per generation attempt. An attempt moves forward through `stage`; `last_successful_stage` survives errors so a
-- failed attempt can resume. Attempts are leased (lease_* columns) so only one worker advances an attempt at a time.
-- A redo, or a retry of a failure that cannot resume in place, is a NEW row pointing at the earlier one
-- (retry_of_attempt_id); earlier rows keep their outcome. The latest attempt of a recipe decides what happens next.
BEGIN;

-- Usage JSON is validated because the cost view casts its fields: token counts are whole numbers, cost_usd is a
-- number or null. (A missing usage object means the call did not run; cost_usd null means it ran but has no rate.)
CREATE FUNCTION catalog_image_usage_is_valid(u jsonb) RETURNS boolean LANGUAGE sql IMMUTABLE AS $$
    SELECT u IS NULL OR (
        jsonb_typeof(u) = 'object'
        AND (u->>'input_tokens' IS NULL OR u->>'input_tokens' ~ '^[0-9]+$')
        AND (u->>'output_text_tokens' IS NULL OR u->>'output_text_tokens' ~ '^[0-9]+$')
        AND (u->>'output_image_tokens' IS NULL OR u->>'output_image_tokens' ~ '^[0-9]+$')
        AND (u->>'reasoning_tokens' IS NULL OR u->>'reasoning_tokens' ~ '^[0-9]+$')
        AND (u->>'cost_usd' IS NULL OR u->>'cost_usd' ~ '^[0-9]+([.][0-9]+)?$'))
$$;

CREATE TABLE catalog_image_generations (
    id                    uuid PRIMARY KEY,
    resource_id           uuid        NOT NULL,                 -- catalog row id (no FK: catalog tables are owned elsewhere)
    resource_type         text        NOT NULL,
    attempt_number        integer     NOT NULL,
    stage                 text        NOT NULL,
    last_successful_stage text        NOT NULL,

    -- What was asked for
    description           text,                                  -- text-model description of the finished dish
    description_version   text,                                  -- version of the description instructions that wrote it
    final_prompt          text,                                  -- exact prompt sent to the image model
    prompt_version        text,                                  -- version of the image prompt template
    settings              jsonb       NOT NULL DEFAULT '{}',     -- models, quality, size, provider

    -- What it cost (tokens are authoritative provider counts; cost_usd is an estimate, null when no rate is known)
    text_usage            jsonb,                                 -- description call(s), incl. repairs and batch share
    image_usage           jsonb,                                 -- image call

    -- Artifacts (their storage keys derive from the attempt id: attempts/<id>/original and attempts/<id>/image.jpg)
    jpeg_sha256           text,
    jpeg_length           bigint,
    image_key             text,                                  -- key returned by the presign API for the uploaded image

    -- Failure and history
    error_type            text,
    error_message         text,                                  -- redacted before it is written
    retry_count           integer     NOT NULL DEFAULT 0,
    retry_of_attempt_id   uuid REFERENCES catalog_image_generations(id),  -- the attempt this one redoes or retries
    redo_note             text,

    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now(),
    completed_at          timestamptz,

    -- Lease: all three are set together when claimed, cleared together when released or retired
    lease_owner           text,
    lease_token           uuid,
    lease_expires_at      timestamptz,

    CONSTRAINT catalog_image_generations_attempt_unique UNIQUE (resource_type, resource_id, attempt_number),
    CONSTRAINT catalog_image_generations_resource_type_check CHECK (resource_type IN ('recipe', 'ingredient')),  -- 'ingredient' reserved
    CONSTRAINT catalog_image_generations_attempt_number_check CHECK (attempt_number > 0),
    CONSTRAINT catalog_image_generations_stage_check CHECK (stage IN (
        'selected', 'described', 'prompt_built', 'generating', 'generated',
        'processed', 'presigned', 'uploaded', 'completed', 'failed',
        'generation_uncertain', 'abandoned')),
    CONSTRAINT catalog_image_generations_last_stage_check CHECK (last_successful_stage IN (
        'selected', 'described', 'prompt_built', 'generated', 'processed',
        'presigned', 'uploaded', 'completed')),
    CONSTRAINT catalog_image_generations_jpeg_length_check CHECK (jpeg_length > 0 AND jpeg_length < 4900000),
    CONSTRAINT catalog_image_generations_image_key_check CHECK (stage NOT IN ('uploaded', 'completed') OR image_key IS NOT NULL),
    CONSTRAINT catalog_image_generations_lease_consistent CHECK (
        (lease_owner IS NULL) = (lease_token IS NULL) AND (lease_token IS NULL) = (lease_expires_at IS NULL)),
    CONSTRAINT catalog_image_generations_usage_valid CHECK (
        catalog_image_usage_is_valid(text_usage) AND catalog_image_usage_is_valid(image_usage))
);

-- At most one in-flight attempt per resource. An uncertain attempt (the image request may have been billed) keeps the
-- resource reserved until it is released. (Lookups by resource and attempt number use the UNIQUE constraint's index.)
CREATE UNIQUE INDEX catalog_image_generations_one_active
    ON catalog_image_generations (resource_type, resource_id)
    WHERE stage IN ('selected', 'described', 'prompt_built', 'generating', 'generated', 'processed', 'presigned', 'uploaded',
                    'generation_uncertain');

-- The candidate scan: open or failed attempts, oldest first.
CREATE INDEX catalog_image_generations_open_by_age
    ON catalog_image_generations (created_at, id)
    WHERE stage IN ('selected', 'described', 'prompt_built', 'generating', 'generated', 'processed', 'presigned', 'uploaded', 'failed');

COMMENT ON COLUMN catalog_image_generations.text_usage IS
    'Description call(s): {provider, model, input_tokens, output_text_tokens, output_image_tokens, reasoning_tokens (already inside output_text_tokens), rates (per 1M tokens, snapshot), cost_usd|null}';
COMMENT ON COLUMN catalog_image_generations.image_usage IS
    'Image call, same shape. cost_usd is an estimate from the rates snapshot; tokens are authoritative.';

-- Tokens and estimated cost per attempt, for reporting. total_cost_usd counts known costs only: when unpriced_calls > 0
-- a call ran but had no rate, so the true cost is higher. A call that did not run (e.g. a redo reusing the description)
-- has no usage and is not counted as unpriced. Example:
--   SELECT date_trunc('day', completed_at) AS day, count(*), sum(total_tokens), sum(total_cost_usd), sum(unpriced_calls)
--   FROM catalog_image_generation_costs WHERE stage = 'completed' GROUP BY 1 ORDER BY 1;
CREATE VIEW catalog_image_generation_costs AS
SELECT
    id AS attempt_id, resource_id, attempt_number, stage, description_version, prompt_version, created_at, completed_at,
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
    (text_usage IS NOT NULL AND text_usage->>'cost_usd' IS NULL)::int
      + (image_usage IS NOT NULL AND image_usage->>'cost_usd' IS NULL)::int  AS unpriced_calls
FROM catalog_image_generations;

COMMIT;
