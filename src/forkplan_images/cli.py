import argparse
import logging
import os
import sys
from contextlib import ExitStack
from uuid import UUID

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from .config import Config
from .descriptions import BatchedDescriber, OpenAIDescriber
from .generation import GeminiGenerator, OpenAIGenerator
from .network import REQUEST_TIMEOUT_SECONDS
from .orchestrator import Options, Orchestrator, Summary
from .pricing import CostLedger
from .redaction import redact
from .repository import PostgresRepository
from .storage import LocalStorage, export_local
from .uploads import HttpUploader


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate missing recipe images with resumable attempts.",
        epilog="Exit codes: 0 success/nothing to do, 1 recipe failure (including uncertain), 2 usage/configuration error.")
    result.add_argument("--limit", type=int, default=10,
                        help="Cap recipes considered; resumed attempts count and run first (default: 10)")
    result.add_argument("--ids", type=UUID, nargs="+", default=(), help="Only these recipe UUIDs")
    result.add_argument("--max-images", type=int, default=1,
                        help="Cap image requests dispatched, including failed/uncertain requests; recovered originals do not count (default: 1)")
    result.add_argument("--dry-run", action="store_true",
                        help="No persistent writes, leases, images, presigns or uploads; text-model calls may incur cost")
    result.add_argument("--batch-descriptions", action="store_true",
                        help="Describe up to 10 recipes per text request (saves text-model requests; results are cached on disk)")
    result.add_argument("-v", "--verbose", action="store_true",
                        help="Log progress (stage changes, skips, failures) to stderr; secrets are redacted")
    result.add_argument("--retry-failures", action="store_true",
                        help="Reopen failed attempts from saved progress; never retry uncertain attempts")
    result.add_argument("--redo", type=UUID, help="Retire previous attempt and regenerate exactly one recipe")
    result.add_argument("--note", help="Optional redo reason (requires --redo)")
    return result


def configure_logging(verbose: bool) -> None:
    """Our own logger only; SDK/HTTP loggers stay silenced in execute() because they can expose URLs."""
    logger = logging.getLogger("forkplan_images")
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.handlers = [handler]
    logger.propagate = False


def execute(config: Config, options: Options, batch_descriptions: bool = False) -> Summary:
    # SDK HTTP logging can expose URLs; only our redacted summaries are emitted.
    for name in ("openai", "httpx", "httpcore", "sqlalchemy"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.handlers = [logging.NullHandler()]
    with ExitStack() as stack:
        engine = create_engine(
            make_url(config.database_url).set(drivername="postgresql+psycopg"),
            hide_parameters=True,
            connect_args={"connect_timeout": 10, "options": "-c statement_timeout=15000 -c lock_timeout=5000"},
        )
        stack.callback(engine.dispose)
        sdk = stack.enter_context(OpenAI(api_key=config.openai_api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0))
        http = stack.enter_context(httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False, trust_env=False))
        storage = LocalStorage(config.artifact_dir)
        describer = OpenAIDescriber(sdk, config.text_model)
        orchestrator = Orchestrator(
            PostgresRepository(engine, config.secrets), storage,
            BatchedDescriber(describer, storage) if batch_descriptions else describer,
            (GeminiGenerator(http, config.gemini_api_key) if config.image_provider == "gemini"
             else OpenAIGenerator(sdk, http)),
            HttpUploader(http, config.api_base_url, config.api_token),
            config.settings, config.lease_seconds, config.secrets,
            export_copy=((lambda recipe, attempt, data: export_local(
                config.local_image_dir, recipe.name, recipe.id, attempt.attempt_number, data))
                if config.local_image_dir else None),
            ledger=CostLedger(config.cost_log),
        )
        return orchestrator.run(options)


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    configure_logging(arguments.verbose)
    try:
        options = Options(
            limit=arguments.limit, ids=tuple(arguments.ids), max_images=arguments.max_images,
            dry_run=arguments.dry_run, retry_failures=arguments.retry_failures,
            redo=arguments.redo, note=arguments.note,
        )
        load_dotenv()
        config = Config.from_environment(os.environ)
    except ValueError as exc:
        print(f"Configuration/usage error: {exc}", file=sys.stderr)
        return 2
    try:
        summary = execute(config, options, arguments.batch_descriptions)
    except Exception as exc:
        # Driver exceptions may contain DSN credentials not recognizable as URLs: the class name is always safe to show,
        # the (redacted) message only with -v.
        print(f"Run failed ({type(exc).__name__}): check database connectivity, migration and configuration. "
              "Re-run with -v for the redacted error.", file=sys.stderr)
        if arguments.verbose:
            detail = redact(str(exc), config.secrets).strip().splitlines()
            print("  " + (detail[0][:300] if detail else "(no message)"), file=sys.stderr)
        return 1
    for result in summary.results:
        print(f"{result.resource_id}: {result.status}" +
              (f" — {redact(result.detail, config.secrets)}" if result.detail else "") +
              (f" [{result.tokens:,} tokens, est. ${result.cost_usd:.4f}]" if result.cost_usd is not None else ""))
    print(f"Recipes considered: {len(summary.results)}; image requests dispatched: {summary.images_dispatched}; "
          f"uncertain: {sum(r.status == 'generation_uncertain' for r in summary.results)}")
    note = f" ({summary.unpriced_calls} call(s) unpriced: no rate configured)" if summary.unpriced_calls else ""
    print(f"Tokens this run: {summary.total_tokens:,}; estimated cost: ${summary.cost_usd:.4f}{note}; usage saved per attempt")
    return summary.exit_code
