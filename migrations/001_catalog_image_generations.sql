-- Apply explicitly, never from the application. Catalog tables remain read-only.
BEGIN;
CREATE TABLE catalog_image_generations (
    id uuid PRIMARY KEY,
    resource_id uuid NOT NULL,
    resource_type text NOT NULL CHECK (resource_type IN ('recipe', 'ingredient')),
    attempt_number integer NOT NULL CHECK (attempt_number > 0),
    description text,
    final_prompt text,
    prompt_version text,
    settings jsonb NOT NULL DEFAULT '{}',
    stage text NOT NULL CHECK (stage IN (
        'selected', 'described', 'prompt_built', 'generating', 'generated',
        'processed', 'presigned', 'uploaded', 'completed', 'failed',
        'generation_uncertain', 'retired'
    )),
    last_successful_stage text NOT NULL CHECK (last_successful_stage IN (
        'selected', 'described', 'prompt_built', 'generated', 'processed',
        'presigned', 'uploaded', 'completed'
    )),
    original_key text,
    jpeg_key text,
    jpeg_sha256 text,
    jpeg_length bigint CHECK (jpeg_length > 0 AND jpeg_length < 4900000),
    image_key text,
    error_type text,
    error_message text,
    retry_count integer NOT NULL DEFAULT 0,
    superseded_by uuid REFERENCES catalog_image_generations(id) DEFERRABLE INITIALLY DEFERRED,
    redo_note text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    lease_owner text,
    lease_token uuid,
    lease_expires_at timestamptz,
    UNIQUE (resource_type, resource_id, attempt_number),
    CHECK ((stage = 'retired') = (superseded_by IS NOT NULL)),
    CHECK (stage NOT IN ('uploaded', 'completed') OR image_key IS NOT NULL)
);
CREATE UNIQUE INDEX catalog_image_generations_one_active
ON catalog_image_generations (resource_type, resource_id)
WHERE stage IN (
    'selected', 'described', 'prompt_built', 'generating',
    'generated', 'processed', 'presigned', 'uploaded'
);
CREATE INDEX catalog_image_generations_resource
ON catalog_image_generations (resource_type, resource_id, attempt_number DESC);
COMMIT;
