import argparse
from contextlib import ExitStack
import logging
import os
import sys
from uuid import UUID

from dotenv import load_dotenv
import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from .config import Config
from .descriptions import OpenAIDescriber
from .generation import OpenAIGenerator
from .orchestrator import Options, Orchestrator, Summary
from .redaction import redact
from .repository import PostgresRepository
from .storage import LocalStorage
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
    result.add_argument("--retry-failures", action="store_true",
                        help="Reopen failed attempts from saved progress; never retry uncertain attempts")
    result.add_argument("--redo", type=UUID, help="Retire previous attempt and regenerate exactly one recipe")
    result.add_argument("--note", help="Optional redo reason (requires --redo)")
    return result


def execute(config: Config, options: Options) -> Summary:
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
        sdk = stack.enter_context(OpenAI(api_key=config.openai_api_key, timeout=60.0, max_retries=0))
        http = stack.enter_context(httpx.Client(timeout=60.0, follow_redirects=False, trust_env=False))
        orchestrator = Orchestrator(
            PostgresRepository(engine, config.secrets), LocalStorage(config.artifact_dir),
            OpenAIDescriber(sdk, config.text_model), OpenAIGenerator(sdk, http),
            HttpUploader(http, config.api_base_url, config.api_token),
            config.settings, config.lease_seconds, config.secrets,
        )
        return orchestrator.run(options)


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
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
        summary = execute(config, options)
    except Exception:
        # Driver exceptions may contain DSN credentials not recognizable as URLs.
        print("Run failed: check database connectivity, migration and configuration.", file=sys.stderr)
        return 1
    for result in summary.results:
        print(f"{result.resource_id}: {result.status}" +
              (f" — {redact(result.detail, config.secrets)}" if result.detail else ""))
    print(f"Recipes considered: {len(summary.results)}; image requests dispatched: {summary.dispatched}; "
          f"uncertain: {sum(r.status == 'generation_uncertain' for r in summary.results)}")
    return summary.exit_code
