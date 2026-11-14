# Recipe image generation CLI

Python 3.11+ MVP for generating missing recipe photographs and uploading them
through the catalog presign API. Catalog tables are **read-only** to this program;
only `catalog_image_generations` is written. No migration is run automatically.

How a recipe flows: the OpenAI text model writes a plain factual description, a fixed
catalogue prompt is built from it, an image model (OpenAI `gpt-image-*` or Gemini) renders
it, the result is converted to a JPEG under 4.9 MB, and it is uploaded through the presign
API. Each step is saved, so a crashed or interrupted run resumes without paying twice.

## Module map

| Module | Responsibility |
| --- | --- |
| `cli.py` | Argument parsing, wiring of real clients, printing the run summary |
| `config.py` | Environment validation into an immutable `Config` |
| `orchestrator.py` | Selects work, enforces the caps, claims attempts, and totals results and cost |
| `attempt_runner.py` | Runs one claimed attempt: one small method per stage, plus failure recording |
| `heartbeat.py` | Renews the lease while slow calls run and detects lost ownership |
| `repository.py`, `state.py` | Postgres persistence, leases and fencing; `Stage`, `Attempt`, `Recipe` |
| `descriptions.py` | Text-model descriptions: single, batched (`BatchedDescriber`) and style checks |
| `prompts.py`, `prompt_templates/` | The image prompt builder; prompt wording lives in plain `.txt` files here |
| `generation.py` | `OpenAIGenerator` and `GeminiGenerator`; definite vs uncertain failures |
| `imaging.py`, `storage.py` | JPEG processing and integrity checks; no-clobber local storage |
| `uploads.py`, `network.py` | Presign and PUT client; bounded retries, timeouts, HTTPS rules |
| `pricing.py` | Token usage, cost estimates (`RATES`) and the CSV cost ledger |
| `redaction.py` | Strips secrets and URLs from anything persisted or printed |

## macOS setup

Install Python 3.11 or newer, then from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,dev]'   # dev adds ruff for linting
cp .env.example .env
```

Edit every placeholder in `.env`. Use a `postgresql+psycopg://` connection URL
(percent-encode special characters in credentials), and an HTTPS API base URL.
Never commit credentials or artifacts. Protect `.env` and the artifact directory
with local permissions; artifact storage is private and is not a public web root.
If you customize `ARTIFACT_DIR` inside this checkout, also add it to `.gitignore`.
`python-dotenv` loads `.env`; already-exported variables take precedence.

Ask your database administrator to review and explicitly apply `migrations/001_catalog_image_generations.sql` once, using a
migration role, before running the CLI. It is a single migration: the tracking table (with its constraints and indexes) and the
`catalog_image_generation_costs` reporting view. It is not safe to re-run; the rollback is in the file's header. The runtime role
needs SELECT on the three catalog tables and SELECT/INSERT/UPDATE on the tracking table, **not** catalog write permissions
(plus SELECT on the view if it will run cost reports).
The external presign API, not this CLI, sets the recipe's `image_key`.

### Configuration

All variables are documented with placeholders in `.env.example`.

| Variable | Policy |
| --- | --- |
| `DATABASE_URL`, `API_BASE_URL`, `OPENAI_API_KEY` | Required |
| `API_TOKEN` and/or the `COGNITO_*` login settings | At least one of the two |
| `OPENAI_TEXT_MODEL` | Required; `gpt-4o-2024-08-06`, `gpt-5.4-mini` or `gpt-5.4-nano` (GPT-5 models are called without a temperature setting) |
| `OPENAI_IMAGE_MODEL` | Required when `IMAGE_PROVIDER=openai`; any `gpt-image-*` ID (e.g. `gpt-image-1.5`); `gpt-image-1` is closed upstream |
| `OPENAI_IMAGE_QUALITY` | OpenAI images only: `low`, `medium`, `high`, `auto`; default `medium` |
| `OPENAI_IMAGE_SIZE` | OpenAI images only: square `1024x1024` only |
| `ARTIFACT_DIR` | Default `artifacts`; store private, durable artifacts here |
| `LOCAL_IMAGE_DIR` | Optional; also saves `<recipe-name>-<id>-a<attempt>.jpg` copies of verified final JPEGs (no-clobber) |
| `IMAGE_PROVIDER` | Optional; `openai` (default) or `gemini`. Text descriptions always use OpenAI |
| `GEMINI_API_KEY`, `GEMINI_IMAGE_MODEL` | Required when `IMAGE_PROVIDER=gemini`; model must look like `gemini-*image*` (e.g. `gemini-3.1-flash-image`) |
| `COST_LOG_PATH` | Optional; default `<ARTIFACT_DIR>/cost_log.csv` (local cost ledger) |
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
forkplan-images --limit 10 --max-images 10 --batch-descriptions   # fewer text requests
forkplan-images --redo RECIPE_UUID --note "Reviewed description; approved a new attempt"
forkplan-images --release-uncertain RECIPE_UUID   # after checking the provider for an uncertain request
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
exception details beyond the error class; add `-v` (or `--log-level debug`) to see the redacted message (for example a missing table). SDK/HTTP request logging is suppressed by the CLI; error
details are redacted before database persistence and output.

## Recommended models

After side-by-side runs (`prompt_experiments/`, gitignored): `OPENAI_IMAGE_MODEL=gpt-image-2.6-flare` (about $0.046 per
medium image, about 10 s) and `OPENAI_TEXT_MODEL=gpt-5.4-mini` (about $0.001 per description). Rates are estimates; confirm
on the OpenAI pricing page. gpt-5.4-mini sometimes describes a curry's chicken as one whole piece, so spot-check curries.

## Batched descriptions (fewer text requests)

`--batch-descriptions` asks the text model for up to 10 recipes' descriptions in one request instead of one request per
recipe, which helps on low text-model request limits. Only the recipes this run can actually generate (up to
`--max-images`) are described. Each description is saved under `ARTIFACT_DIR/descriptions/` as soon as the batch returns,
so rerunning never repeats a paid request. A recipe the batch missed or that used marketing wording is described by a
normal single request. The batch's cost is split evenly across its recipes in the ledger.

## Prompt text and versions

The wording of the image style, description instructions and batch suffix is in
`src/forkplan_images/prompt_templates/*.txt` (hand-wrapped lines are joined on load). Stored attempts and cached
descriptions are keyed by `PROMPT_VERSION` / `DESCRIPTION_VERSION`, so a wording change must come with a version bump.
`tests/test_prompt_versions.py` pins a hash of each text to its version and fails until you bump the version and update
the hash.

Description rules live in the instructions *and* are enforced in code: `STYLE_BLOCKLIST` rejects marketing, taste and
mood words (perfect, glossy, vibrant, tender, savory, comforting, appealing, rich, fresh, and similar) and descriptions outside
200-700 characters. A rejected description gets one
repair call that sends the previous text back to be edited; if it still contains a banned word the recipe fails (re-run
with `--retry-failures`). Recipes are sent to the text model as `<recipe>` blocks, and the image prompt opens with the
dish name so the image model uses its own knowledge of how the dish is served.

### Signing in to the upload API with a Cognito service account

Instead of (or as a backup for) a fixed `API_TOKEN`, which expires, the program can sign in as a dedicated Cognito user
and renew the token itself. Set these:

```
COGNITO_REGION=eu-west-2
COGNITO_CLIENT_ID=<app client id>
COGNITO_USERNAME=<service account username>
COGNITO_PASSWORD=<its password>
# COGNITO_CLIENT_SECRET=<only if the app client has a secret>
# COGNITO_TOKEN_USE=access     # or "id", whichever token your backend validates (default: access)
```

**Both set:** `API_TOKEN` is used first. If the API rejects it, the program signs in with the service account and stays on
that login for the rest of the run. **Only `COGNITO_*` set:** it logs in when it first needs a token, keeps it until a minute before it expires, and if the API still answers 401 it
signs in again and retries once. It does not use refresh tokens, so there is nothing rotating to store and several
Lambdas can run at once. A refused login stops the run and says why (wrong password, flow not enabled, and so on).

One-time setup on the Cognito side:

1. Create the user and give it a **permanent** password (otherwise login returns a `NEW_PASSWORD_REQUIRED` challenge):
   `aws cognito-idp admin-create-user ...` then
   `aws cognito-idp admin-set-user-password --user-pool-id ... --username ... --password ... --permanent`
2. On the app client, enable the **`ALLOW_USER_PASSWORD_AUTH`** auth flow.
3. Give the user whatever group or scope your backend needs to allow image presigning for catalog recipes.

In a Lambda, put the password in Secrets Manager (not the function's plain environment) and load it into
`COGNITO_PASSWORD` when the function starts. Never commit it.

### Staying under OpenAI rate limits

Set the limits for your account in `.env` (both optional; `0` or unset means no pacing):

```
OPENAI_IMAGE_RPM=8    # image requests per minute
OPENAI_TEXT_RPM=20    # text (description) requests per minute
```

The program spaces requests evenly (`8` means one every 7.5 seconds, never more than 8 in any minute) and waits **before**
an image request is recorded as sent, so a wait can never be mistaken for an uncertain request. Waiting keeps the lease alive.
Use a value a little under your real limit (see the OpenAI limits page for your tier). One run's pace is about 6 images a
minute, so pacing only slows it when the limit is below that. If OpenAI still answers 429, the run stops
(like rejected credentials), keeps saved work and tells you to lower the setting or wait and run `--retry-failures`.
The limiter works inside one run; two runs at the same time each pace themselves, so use a lower value or run one at a time.

### Rejected credentials (HTTP 401/403) and rate limits (429)

If any service (OpenAI, the image provider or your upload API) rejects the credentials, the run **stops immediately**
instead of trying the next recipe, because every recipe would fail the same way and each would still pay for an image
before its upload failed. The message says what to fix. Nothing is lost: the failed attempt keeps its saved image, and
after you fix the token `forkplan-images --retry-failures --max-images 0 --limit 1` resumes it from where it stopped
and uploads that image without generating a new one.

## Logging

Logs go to stderr. Choose what you see with `--log-level debug|info|warning|error` or the `LOG_LEVEL` environment
variable (the option wins; `-v` is shorthand for `--log-level debug` and also shows the redacted message when a run
fails). The default is `info`.

| Level | What you see |
|---|---|
| `debug` | Everything below, plus lease renewals, saved batch descriptions being reused and other internals |
| `info` (default) | Progress: run settings, how many recipes were selected, `[3/10]` per recipe, each stage (described, prompt built, requesting the image, generated, processed, presigned, uploaded, completed), reused descriptions, description edits, tokens and estimated cost per recipe, and a run summary |
| `warning` | Problems only: failed or uncertain attempts, lost leases, batch fallbacks, ledger write failures. A clean run prints nothing |
| `error` | Only things that could not be recorded, such as progress that failed to save |

Messages carry short IDs, recipe names and redacted error text, never prompts, URLs or credentials. SDK and HTTP
library logging stays silenced because it can expose URLs.

## Tokens and cost per recipe

Every description and image call records the provider's token counts and an estimated cost, in three places:

* **Printed after each run:** `<recipe id>: completed [3,209 tokens, est. $0.0505]` per recipe, plus `Tokens this run: ...; estimated cost: ...`.
* **Local CSV ledger,** `cost_log.csv` in `ARTIFACT_DIR` (override with `COST_LOG_PATH`): one row per recipe per run that made a billable
  call (dry runs too, because their text calls cost money). Columns: text and image model, input / output / reasoning tokens, per-call
  cost, `text_total_tokens`, `image_total_tokens`, `total_tokens`, `total_cost_usd`, prompt version and status. A ledger written by an older
  version is upgraded in place (old rows keep blanks in the new columns). A full disk or unwritable path never fails or retries a generation.
* **Database,** per attempt, in `text_usage` / `image_usage` (jsonb), and a read-only view,
  `catalog_image_generation_costs`, with one row per attempt: token counts by call, `total_tokens`, `text_cost_usd`, `image_cost_usd`
  and `total_cost_usd`. Example: `SELECT sum(total_tokens), sum(total_cost_usd) FROM catalog_image_generation_costs;`

* **Tokens are authoritative; dollars are estimates** from `src/forkplan_images/pricing.py`. Confirm the rates on the provider pricing
  pages and edit `RATES` when they change. A model with no known rate, or an unpriced token class, records `cost_usd: null` and is
  counted as "unpriced" in the run total and in the view's `unpriced_calls` column (`total_cost_usd` then covers known costs only; a call that did not run has no usage and is not unpriced).
* **Reasoning tokens** (GPT-5 text models) are recorded separately for visibility but are already part of the output tokens and the cost;
  they are not counted twice in `total_tokens`.
* **Batched descriptions:** one request covers up to 10 recipes, so its tokens and cost are split evenly across them. Descriptions loaded
  from the on-disk cache cost nothing in later runs.
* A call that fails before a usable response (timeouts, 4xx/5xx, blocked prompts) has no usage to record;
  an uncertain attempt may still have been billed. Check the provider dashboard for those.
* Gemini image models return about 1,120 image-output tokens per 1K image, plus a few hundred text/reasoning tokens.

## State, recovery and concurrency

Attempts progress through:

```
selected → described → prompt_built → generating → generated
         → processed → presigned → uploaded → completed
```

`last_successful_stage` survives errors separately from `stage`. `failed`,
`failed`, `generation_uncertain`, `completed`, and `abandoned` are terminal. Fresh selection
requires a missing/empty recipe image, PUBLISHED/DRAFT status, and **no attempt at all**, even if the recipe
image is subsequently cleared. Resume comes from tracking state, not the recipe's possibly premature image key.
`completed` means the JPEG was uploaded to storage; publishing it on the recipe is the main system's job.

History is never rewritten: a redo or retry is a **new row** whose `retry_of_attempt_id` points at the earlier one,
and the earlier row keeps its outcome (a completed attempt stays completed). The latest attempt decides what happens next.

* `--retry-failures` looks at recipes whose **latest** attempt failed. If it failed after the image was saved
  (`generated` or later) the same attempt resumes, clears the error and increments `retry_count`. If it failed earlier,
  the next step would be a new paid image request, so a **new attempt** is created instead (it reuses the saved description).
* Automatic retries are capped: `--retry-failures` skips a recipe once it has 4 attempts (or one attempt has been
  resumed 4 times). `--redo` is the manual override.
* `--redo ID` creates a new attempt using the CLI's current settings and links it to the previous one. The description is
  reused (no text call, no text usage) when it came from the current description version. An unfinished attempt with an
  expired lease is set aside as `abandoned`; a valid lease refuses redo. It also works for a recipe with an existing image
  and no tracking history.
* `generation_uncertain` (the image request may have been billed) reserves the recipe: it is never retried and `--redo`
  is refused. After checking the provider, run `--release-uncertain RECIPE_ID` to set it aside (stage `abandoned`), then redo.
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

Text calls (except HTTP 429), downloads, presign and PUT use at most three attempts with
exponential backoff and jitter. A text-model 429 is not retried: an immediate retry is also
rate limited and spends more of the quota. Re-run with `--retry-failures` instead. Lease checkpoints fence each retry, so losing ownership cannot
dispatch another request. Each network call has a timeout.
Database connect/statement/lock timeouts are bounded too.

## Repository extras

* `data/world-cuisines-menu.csv` is a 40-dish sample menu (Indian, Japanese, Korean, Italian) with my own estimated
  macros. The CLI does **not** read it: it only reads recipes from the catalog database.
* `local_images/` and `prompt_experiments/` hold local output and prompt experiments; both are gitignored.

## Small manual pilot (explicitly paid, outside the tests)

1. Run `--dry-run --limit 3`. Text calls may incur cost. Review the selected
   recipes and descriptions in an approved local debugging session if needed;
   the CLI deliberately prints statuses, not recipe prompts or signed URLs.
2. Run `--limit 1 --max-images 1`. Inspect the tracking row's description and
   local image, then verify the public image through your catalog application.
3. Check ingredient accuracy **by eye**, in both the description and photo:
   no invented sides/toppings/garnishes; portions fit the quantities; a
   consistent 45-degree three-quarter angle, a single neutral ceramic plate, white wood and soft light,
   and no cutlery, cups, napkins or condiments in frame.
4. If acceptable, run `--limit 5 --max-images 5` and repeat the checks. Use
   `--redo ID --note ...` only after reviewing uncertain billing or completed
   results. Never mass-retry uncertain attempts.

Description validation checks typed schema, 20–2000 character bounds and
refusal/apology language. Ingredient fidelity is instructed in the description
prompt and checked in this pilot, not by requiring visible ingredient names.

## Tests

```sh
pytest
ruff check src tests   # needs the dev extra
```

Postgres integration tests (`tests/integration/`) are skipped unless `TEST_DATABASE_URL` points at a throwaway database. They create a
temporary schema, apply every migration to it and exercise the real SQL (claims, lease expiry, retry, redo, constraints, the cost view):

```sh
docker run -d --name fp-pg -e POSTGRES_PASSWORD=testpass -e POSTGRES_DB=catalog -p 55432:5432 postgres:16-alpine
TEST_DATABASE_URL=postgresql+psycopg://postgres:testpass@localhost:55432/catalog pytest tests/integration
```

Tests mock OpenAI/HTTP and use a repository fake, **not SQLite**. A suite-wide
socket guard rejects real network/database connections. Real SDK parsing and
request serialization are exercised with mocked HTTP. No live image call or
database is needed.

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
* **Real Postgres:** the integration tests above verify the migrations and repository SQL on Postgres 16 (found and fixed an empty `--ids`
  query bug). Still verify in your environment: migration compatibility, the partial unique index,
  advisory locking under competing claims/redo, transaction commits before
  generation, and single-statement token/expiry fencing (including a row-lock
  wait crossing lease expiry). The mocked SQL tests cannot prove these guarantees.
  Verify runtime grants and the chosen database timeouts in your environment.
* **API/S3:** verify auth/scopes, presign response/recipe side effects, content
  length/type, expiry, exact-byte PUTs, lack of bearer auth on S3, repeated PUTs,
  newer presign keys after interruption, and public catalog visibility.
* **Gemini:** the request shape, usage fields and the flash/pro/lite rates in `pricing.py` are not verified
  against official documentation (pro and lite rates are third-party figures); confirm before relying on cost totals.
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
