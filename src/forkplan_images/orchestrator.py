import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from .attempt_runner import AttemptRunner, ExportCopy, Services
from .descriptions import BatchedDescriber, Describer, validate_description
from .generation import Generator
from .heartbeat import Heartbeat
from .network import blocking_message, blocking_status
from .pricing import CostLedger, total_cost, total_tokens
from .prompts import build_prompt
from .ratelimit import RateLimiter
from .redaction import redact
from .repository import Repository, Work
from .state import Attempt, ClaimConflict, LeaseLost, Stage
from .storage import Storage
from .uploads import Uploader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Options:
    limit: int = 10
    ids: tuple[UUID, ...] = ()
    max_images: int = 1
    dry_run: bool = False
    retry_failures: bool = False
    redo: UUID | None = None
    note: str | None = None
    release_uncertain: UUID | None = None

    def __post_init__(self) -> None:
        if self.limit < 0 or self.max_images < 0:
            raise ValueError("Caps must be nonnegative")
        if self.release_uncertain and (self.redo or self.dry_run or self.ids):
            raise ValueError("Release cannot be combined with redo, dry run or IDs")
        if self.redo and (self.dry_run or self.ids):
            raise ValueError("Redo cannot be combined with dry run or IDs")
        if self.note and not self.redo:
            raise ValueError("Note requires redo")


@dataclass(frozen=True)
class Result:
    resource_id: UUID
    status: str
    detail: str = ""
    cost_usd: float | None = None  # spend incurred by THIS run; None when nothing was billed/priced
    tokens: int | None = None  # tokens used by THIS run (text + image); None when no call reported usage


@dataclass
class Summary:
    results: list[Result] = field(default_factory=list)
    images_dispatched: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    total_tokens: int = 0
    stopped_because: str | None = None  # set when the run ended early, e.g. rejected credentials

    def count_image_dispatch(self) -> None:
        self.images_dispatched += 1

    @property
    def exit_code(self) -> int:
        return int(bool(self.stopped_because) or any(r.status in ("failed", "generation_uncertain", "lease_lost")
                                                     for r in self.results))


class Orchestrator:
    def __init__(self, repository: Repository, storage: Storage, describer: Describer,
                 generator: Generator, uploader: Uploader, settings: dict[str, Any],
                 lease_seconds: int = 300, secrets: tuple[str, ...] = (),
                 export_copy: ExportCopy | None = None, image_limiter: RateLimiter | None = None,
                 ledger: CostLedger | None = None):
        self.repository, self.storage = repository, storage
        self.describer, self.generator, self.uploader = describer, generator, uploader
        self.settings, self.lease_seconds, self.secrets = settings, lease_seconds, secrets
        self.export_copy = export_copy
        self.image_limiter = image_limiter
        self.ledger = ledger

    def run(self, options: Options) -> Summary:
        summary = Summary()
        if not options.limit:
            return summary
        owner = str(uuid4())
        log.info("Run started: limit=%d max_images=%d dry_run=%s retry_failures=%s | %s", options.limit,
                 options.max_images, options.dry_run, options.retry_failures,
                 ", ".join(f"{k}={v}" for k, v in self.settings.items()))
        work = self._select_work(options, owner, summary)
        log.info("Selected %d recipe(s): %d resuming, %d retrying after a failure, %d new", len(work),
                 sum(w.attempt is not None for w in work), sum(w.retry_of is not None for w in work),
                 sum(w.attempt is None and w.retry_of is None for w in work))
        self._prefetch_descriptions(work, options)
        for number, item in enumerate(work, 1):
            log.info("[%d/%d] %s", number, len(work), self._plan(item))
            if options.dry_run:
                summary.results.append(self._dry_run(item, summary))
            else:
                self._run_item(item, options, owner, summary)
            if summary.stopped_because:
                log.error("Stopping the run: %s", summary.stopped_because)
                break
        log.info("Run finished: %d recipe(s), %d image request(s), %s tokens, est. $%.4f", len(summary.results),
                 summary.images_dispatched, f"{summary.total_tokens:,}", summary.cost_usd)
        return summary

    @staticmethod
    def _plan(item: Work) -> str:
        label = f"'{item.recipe.name}' ({str(item.recipe.id)[:8]})"
        if item.attempt:
            return f"{label}: resuming attempt {item.attempt.attempt_number} from {item.attempt.stage.value.replace('_', ' ')}"
        if item.retry_of:
            return f"{label}: new attempt {item.retry_of.attempt_number + 1}, retrying attempt {item.retry_of.attempt_number}"
        return f"{label}: new recipe"

    def _select_work(self, options: Options, owner: str, summary: Summary) -> list[Work]:
        """Resumable and fresh recipes, or the single redo attempt. A refused redo is reported and yields no work."""
        if options.release_uncertain:
            try:
                self.repository.release_uncertain(options.release_uncertain)
                summary.results.append(Result(options.release_uncertain, "released",
                                              "Uncertain attempt released; it can now be redone"))
            except ClaimConflict as exc:
                summary.results.append(Result(options.release_uncertain, "failed", str(exc)))
            return []
        if not options.redo:
            return self.repository.candidates(options.limit, options.ids, options.retry_failures)
        if not options.max_images:
            summary.results.append(Result(options.redo, "skipped", "Image cap; no writes"))
            return []
        try:
            return [self.repository.redo(options.redo, owner, self.lease_seconds, options.note)]
        except ClaimConflict as exc:
            summary.results.append(Result(options.redo, "failed", str(exc)))
            return []

    def _dry_run(self, item: Work, summary: Summary) -> Result:
        """Report what a real run would do. Only the (billed) description call is made; nothing is persisted."""
        try:
            if item.attempt:
                return Result(item.recipe.id, "dry_run", f"Would resume from {item.attempt.stage}")
            if item.retry_of:
                return Result(item.recipe.id, "dry_run", "Would start a new attempt (no text call: reuses the saved description)")
            try:
                description = validate_description(self.describer.describe(item.recipe))
            finally:
                usage = getattr(self.describer, "last_usage", None)
                if isinstance(usage, dict):
                    self._add_to_summary(summary, usage)
                    self._record_in_ledger(run_kind="dry_run", recipe_id=item.recipe.id,
                                           recipe_name=item.recipe.name, status="dry_run", text=usage)
            build_prompt(item.recipe, description)
            return Result(item.recipe.id, "dry_run", "Description validated; would generate and upload")
        except Exception as exc:
            if blocked := blocking_status(exc):
                summary.stopped_because = blocking_message(exc, blocked)
            return Result(item.recipe.id, "failed", redact(str(exc), self.secrets))

    def _run_item(self, item: Work, options: Options, owner: str, summary: Summary) -> None:
        stage = item.attempt.stage if item.attempt else Stage.SELECTED
        if stage == Stage.FAILED:
            stage = item.attempt.last_successful_stage
        needs_generation = stage in (Stage.SELECTED, Stage.DESCRIBED, Stage.PROMPT_BUILT)
        if needs_generation and summary.images_dispatched >= options.max_images:
            log.info("%s: skipped (image cap reached)", item.recipe.name)
            summary.results.append(Result(item.recipe.id, "skipped", "Image cap; no writes"))
            return
        try:
            attempt = (item.attempt if options.redo else
                       self.repository.claim(item, owner, self.lease_seconds))
        except ClaimConflict:
            log.info("%s: claim conflict (another worker has it); skipped", item.recipe.name)
            summary.results.append(Result(item.recipe.id, "skipped", "Claim conflict"))
            return
        except Exception as exc:
            summary.results.append(Result(item.recipe.id, "failed", redact(str(exc), self.secrets)))
            return
        self._process(item, attempt, summary)

    @staticmethod
    def _add_to_summary(summary: Summary, *usages: dict[str, Any]) -> float:
        known, unpriced = total_cost(*usages)
        summary.cost_usd += known
        summary.unpriced_calls += unpriced
        summary.total_tokens += total_tokens(*usages)
        return known

    def _record_in_ledger(self, **row: Any) -> None:
        if self.ledger:
            try:
                self.ledger.record(**row)
            except OSError as exc:  # a full disk must never fail or retry a generation
                log.warning("cost ledger write failed: %s", type(exc).__name__)

    def _prefetch_descriptions(self, work: list[Work], options: Options) -> None:
        """Batch the descriptions this run will need, up to the image cap, in as few requests as possible."""
        if not isinstance(self.describer, BatchedDescriber):
            return
        pending = [w.recipe for w in work
                   if (w.attempt is None and not (w.retry_of and w.retry_of.description))
                   or (w.attempt and w.attempt.stage == Stage.SELECTED and not w.attempt.description)]
        cap = None if options.dry_run else options.max_images
        try:
            self.describer.prefetch(pending[:cap], persist=not options.dry_run)
        except Exception as exc:  # unprefetched recipes are described one by one and report their own errors
            log.warning("batched descriptions failed (%s); describing recipes one by one",
                        redact(str(exc), self.secrets))

    def _process(self, work: Work, attempt: Attempt, summary: Summary) -> None:
        token = attempt.lease_token
        try:
            with Heartbeat(self.repository, attempt, self.lease_seconds) as heartbeat:
                runner = AttemptRunner(self._services(), work, attempt, heartbeat, summary.count_image_dispatch)
                outcome = runner.run()
                attempt = runner.attempt
                if outcome.stop_run:
                    summary.stopped_because = outcome.detail
                usages = list(runner.usage.values())
                known = self._add_to_summary(summary, *usages)
                tokens = total_tokens(*usages)
                summary.results.append(Result(work.recipe.id, outcome.status, outcome.detail,
                                              round(known, 6) if usages else None, tokens if usages else None))
                if usages:
                    self._record_in_ledger(
                        run_kind="generate", recipe_id=work.recipe.id, recipe_name=work.recipe.name,
                        attempt_id=attempt.id, attempt_number=attempt.attempt_number, status=outcome.status,
                        prompt_version=attempt.prompt_version or "",
                        text=runner.usage.get("text"), image=runner.usage.get("image"))
                log.info("%s: %s%s", work.recipe.name, outcome.status,
                         f" ({tokens:,} tokens, est. ${known:.4f})" if usages else "")
        except LeaseLost:
            # Never retry a rejected write, including error or release writes.
            log.warning("recipe %s: ownership lost; stopped", work.recipe.id)
            summary.results.append(Result(work.recipe.id, "lease_lost", "Ownership lost; stopped"))
            return
        except Exception as exc:
            # A database/storage outage may prevent even recording an error.
            detail = redact(str(exc), self.secrets)
            log.error("recipe %s: could not record progress: %s", work.recipe.id, detail)
            summary.results.append(Result(work.recipe.id, "failed", detail))
            return
        try:
            self.repository.release(attempt.id, token)
        except Exception:
            # Never retry release: another worker may already own the attempt.
            log.warning("recipe %s: lease release rejected; leaving it to expire", work.recipe.id)

    def _services(self) -> Services:
        return Services(self.repository, self.storage, self.describer, self.generator, self.uploader,
                        self.settings, self.secrets, self.export_copy, self.image_limiter)
