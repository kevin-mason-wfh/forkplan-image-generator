# Recipe image generation CLI

Python 3.11+ MVP for generating missing recipe photographs and uploading them
through the catalog presign API. Catalog tables are **read-only** to this program;
only `catalog_image_generations` is written. No migration is run automatically.

## macOS setup

Install Python 3.11 or newer, then from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
```

Edit every placeholder in `.env`. Use a `postgresql+psycopg://` connection URL
(percent-encode special characters in credentials), and an HTTPS API base URL.
Never commit credentials or artifacts. Protect `.env` and the artifact directory
with local permissions; artifact storage is private and is not a public web root.
If you customize `ARTIFACT_DIR` inside this checkout, also add it to `.gitignore`.
`python-dotenv` loads `.env`; already-exported variables take precedence.

Ask your database administrator to review and explicitly apply
`migrations/001_catalog_image_generations.sql`, using a migration role, before
running the CLI. The runtime role needs SELECT on the three catalog tables and
SELECT/INSERT/UPDATE on the tracking table, **not** catalog write permissions.
The external presign API, not this CLI, sets the recipe's `image_key`.

### Configuration

All variables are documented with placeholders in `.env.example`.

| Variable | Policy |
| --- | --- |
| `DATABASE_URL`, `API_BASE_URL`, `API_TOKEN`, `OPENAI_API_KEY` | Required |
| `OPENAI_TEXT_MODEL` | Required; `gpt-4o-2024-08-06` |
| `OPENAI_IMAGE_MODEL` | Required; `gpt-image-1`, `gpt-image-1-mini`, `gpt-image-1.5` |
| `OPENAI_IMAGE_QUALITY` | `low`, `medium`, `high`, `auto`; default `medium` |
| `OPENAI_IMAGE_SIZE` | Square `1024x1024` only |
| `ARTIFACT_DIR` | Default `artifacts`; store private, durable artifacts here |
| `LEASE_SECONDS` | Default 300; must exceed the 180-second generation timeout |

Model names have **no defaults**. The explicit allow-lists are deliberately
narrower than the API. Expand them only after checking official documentation.
The SDK is pinned to an `httpx`-based release so `respx` can mock its transport.
Typed output schemas use Pydantic, already required by the official OpenAI SDK;
there are no extra framework/runtime dependencies.

## Commands

```sh
forkplan-images --help
forkplan-images --dry-run --limit 3
forkplan-images --limit 1 --max-images 1
forkplan-images --ids RECIPE_UUID OTHER_RECIPE_UUID --max-images 2
forkplan-images --retry-failures --limit 5 --max-images 1
forkplan-images --redo RECIPE_UUID --note "Reviewed description; approved a new attempt"
# Equivalent entry point:
python -m forkplan_images --help
```

`--limit` caps recipes **considered**, including resumed work, which comes first.
`--max-images` caps generation **requests dispatched**, including requests that
fail or become uncertain. Recovered originals cost no new request and do not
count. Defaults are 10 considered recipes and 1 dispatched image. Either cap can
be zero. Work skipped by a cap is untouched: no row, lease or failure is written.
Work already past generation can still complete when the image cap is exhausted.

**Dry run is not necessarily free:** it can call the text model for fresh
recipes. It makes no tracking writes, leases, artifact writes, image-generation
calls, presign requests or uploads. Existing attempts are reported without being
modified. The image cap applies to actual dispatches, so it does not limit
dry-run text calls; use `--limit` for that. `--redo --dry-run` and
`--redo --ids ...` are refused. `--note` requires `--redo`.

Each considered recipe gets a summary, followed by dispatch and uncertainty
counts. Exit codes: **0** success/nothing to do (including ordinary claim
conflicts or capped work), **1** any recipe failure, uncertainty, lease loss or
refused redo, **2** usage/configuration error. A failed recipe does not stop
other recipes. Runtime connection failures return 1 without exposing driver
exception details. SDK/HTTP request logging is suppressed by the CLI; error
details are redacted before database persistence and output.

## State, recovery and concurrency

Attempts progress through:

```
selected → described → prompt_built → generating → generated
         → processed → presigned → uploaded → completed
```

`last_successful_stage` survives errors separately from `stage`. `failed`,
`generation_uncertain`, `completed`, and `retired` are terminal. Fresh selection
requires a missing/empty recipe image and PUBLISHED/DRAFT status, and excludes
**every non-retired attempt**, even if the recipe image is subsequently cleared.
Resume comes from tracking state, not the recipe's possibly premature image key.

* `--retry-failures` reopens the same failed attempt after its last successful
  stage, clears errors and increments the retry count. It never reopens uncertain
  attempts.
* `--redo ID` atomically retires the previous non-retired attempt, records its
  successor, and creates a new attempt with a higher number. It also works for a
  recipe with an existing image and no tracking history. A valid lease refuses
  redo. All history and originals are retained.
* A partial unique index enforces one **active** attempt per resource.
  Transaction-scoped advisory locks serialize fresh claims and redo without
  locking or writing catalog rows. Colliding claims are skipped.
* Every claim issues a new token. Every state/artifact/error/completion write,
  renewal and release checks that token and expiry in the UPDATE itself.
  Mutations acquire the row lock first, then use database `clock_timestamp()`
  (not transaction-start `now()` or client clocks) to prevent lock waits from
  resurrecting an expired lease.
* A heartbeat renews every third of the lease duration, including during
  generation and upload. Ownership is also renewed before slow operations.
  If renewal/ownership fails, the worker stops without retrying rejected writes.
  An already in-flight synchronous network request cannot be recalled; its
  result is discarded on ownership loss and no subsequent stage is performed.
* `generating` is committed **before** the image call. Image SDK automatic
  retries are disabled. Timeout, transport failure, 5xx, HTTP 408, response decode
  or image-download failure are conservatively uncertain. Rejected 4xx,
  including 429, are definite failures with no automatic generation retry.
* Originals use `attempts/{attempt_id}/original`, without relying on a recorded
  path or format extension. Original response bytes are saved untouched before
  `generated`. If interrupted at `generating`, an existing original is recovered;
  otherwise the attempt becomes uncertain, with no regeneration.
* Local publication writes a temporary file, fsyncs it, then atomically **links**
  it into place and removes the temporary name. This intentionally uses a
  no-clobber hard link rather than POSIX rename, which could overwrite an
  original. Existing keys are errors. Temporary files are never used for recovery.
* JPEG encoding reduces quality, then dimensions, until strictly below
  4,900,000 bytes. Original bytes remain unchanged; JPEG hash/length are stored.
* From `processed` or `presigned`, verify the saved JPEG's SHA-256 and length,
  request a **fresh presign**, then PUT the saved bytes. Only the API receives
  the bearer token. The PUT uses the returned content type and no Authorization.
  Redirects are not followed. HTTPS is required.
* A successful PUT persists `uploaded` and **that presign's image key together**.
  Recovery at `uploaded` only completes: no new presign or PUT. A crash between
  PUT and the `uploaded` write causes a new presign/PUT using the saved JPEG,
  with the newer key recorded. Upload failures never regenerate an image.

Text calls, downloads, presign and PUT use at most three attempts with exponential
backoff and jitter. Lease checkpoints fence each retry, so losing ownership cannot
dispatch another request. Each network call has a timeout.
Database connect/statement/lock timeouts are bounded too.

## Small manual pilot (explicitly paid, outside the tests)

1. Run `--dry-run --limit 3`. Text calls may incur cost. Review the selected
   recipes and descriptions in an approved local debugging session if needed;
   the CLI deliberately prints statuses, not recipe prompts or signed URLs.
2. Run `--limit 1 --max-images 1`. Inspect the tracking row's description and
   local image, then verify the public image through your catalog application.
3. Check ingredient accuracy **by eye**, in both the description and photo:
   no invented sides/toppings/garnishes; portions fit the quantities; consistent
   overhead photography, neutral ceramic tableware, white wood and soft light.
4. If acceptable, run `--limit 5 --max-images 5` and repeat the checks. Use
   `--redo ID --note ...` only after reviewing uncertain billing or completed
   results. Never mass-retry uncertain attempts.

Description validation checks typed schema, 20–2000 character bounds and
refusal/apology language. Ingredient fidelity is instructed in the description
prompt and checked in this pilot, not by requiring visible ingredient names.

## Tests

```sh
pytest
```

Tests mock OpenAI/HTTP and use a repository fake, **not SQLite**. A suite-wide
socket guard rejects real network/database connections. Real SDK parsing and
request serialization are exercised with mocked HTTP. No live image call or
database is needed. No pre-existing lint/build tools were present.

### Assumptions and remaining integration checks

* Resource scope is recipes only; `resource_type` reserves ingredients.
* Recipe/ingredient UUIDs and the specified schema already exist. Quantities and
  nullable fields are passed as data without inventing missing values.
* `--ids` filters resumed and fresh work. Redo deliberately bypasses the
  missing-image/status filter, allowing explicit replacement of an existing image.
* The artifact directory is durable on the same machine across runs and is
  trusted/private (not writable by other local users). All artifact keys are
  relative and backend-independent.
* Presign success is assumed to return the documented fields, HTTPS upload URL
  and JPEG content type. A lost presign response may change the recipe key; the
  next presign safely replaces it. Public URLs are not persisted.
* **Real Postgres:** verify migration compatibility, the partial unique index,
  advisory locking under competing claims/redo, transaction commits before
  generation, and single-statement token/expiry fencing (including a row-lock
  wait crossing lease expiry). The mocked SQL tests cannot prove these guarantees.
  Verify runtime grants and the chosen database timeouts in your environment.
* **API/S3:** verify auth/scopes, presign response/recipe side effects, content
  length/type, expiry, exact-byte PUTs, lack of bearer auth on S3, repeated PUTs,
  newer presign keys after interruption, and public catalog visibility.
* **OpenAI:** parameters below were checked in official SDK documentation/types;
  account access, model availability, actual latency/billing, text/image quality
  and ingredient fidelity remain live pilot checks. No paid verification was
  performed. Hosted OpenAI guides could not be reached; arbitrary resolutions
  and newer model-specific quality values were not verified and are rejected.

Official sources:
[typed parsing](https://github.com/openai/openai-python/blob/718aee979a9c334d37302aeff21cee6cbff4d43c/helpers.md),
[image parameters](https://github.com/openai/openai-python/blob/718aee979a9c334d37302aeff21cee6cbff4d43c/src/openai/types/image_generate_params.py),
[image responses](https://github.com/openai/openai-python/blob/718aee979a9c334d37302aeff21cee6cbff4d43c/src/openai/types/image.py),
[SDK retries](https://github.com/openai/openai-python#retries).
The supported GPT Image models use base64 PNG output with `n=1`; no unsupported
`response_format` or legacy `style` parameter is sent. URL responses also have
an isolated, timeout-bound download path.

## Later: serverless migration (not implemented)

Keep the dependency-injected orchestrator and Postgres repository. Replace
`LocalStorage` with a private S3 backend implementing `put/get/exists`, including
conditional no-overwrite writes. Use Step Functions **Standard** for durable
workflow coordination and Lambda for bounded steps, maintaining the same
committed generation marker, fencing tokens and artifact-first recovery.
Do not let workflow retries blindly repeat image generation. Store credentials
in a managed secret store, use private S3 artifacts and least-privilege IAM,
and account for Lambda execution limits and lease heartbeats. No scheduler,
deployment resources or cloud infrastructure are included in this MVP.
