from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, Thread
from typing import Any, TypeVar
from uuid import UUID, uuid4

from .descriptions import Describer, validate
from .generation import GenerationError, Generator
from .imaging import fingerprint, process, verify
from .prompts import PROMPT_VERSION, build_prompt
from .redaction import redact
from .repository import Repository, Work
from .state import Attempt, ClaimConflict, LeaseLost, Stage
from .storage import Storage
from .uploads import Uploader


T = TypeVar("T")


class Heartbeat:
    def __init__(self, repository: Repository, attempt: Attempt, seconds: int):
        self.repository, self.attempt, self.seconds = repository, attempt, seconds
        self.stop = Event()
        self.lost = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.seconds / 3):
            try:
                self.repository.renew(self.attempt.id, self.attempt.lease_token, self.seconds)
            except Exception:
                self.lost.set()
                return

    def check(self) -> None:
        if self.lost.is_set():
            raise LeaseLost("Lease renewal failed; processing stopped")

    def call(self, operation: Callable[[], T]) -> T:
        self.check()
        try:
            self.repository.renew(self.attempt.id, self.attempt.lease_token, self.seconds)
        except Exception:
            self.lost.set()
            raise LeaseLost("Lease renewal failed; processing stopped") from None
        self.check()
        try:
            result = operation()
        finally:
            self.check()
        self.check()
        return result

    def __enter__(self) -> "Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        self.thread.join()


@dataclass(frozen=True)
class Options:
    limit: int = 10
    ids: tuple[UUID, ...] = ()
    max_images: int = 1
    dry_run: bool = False
    retry_failures: bool = False
    redo: UUID | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.limit < 0 or self.max_images < 0:
            raise ValueError("Caps must be nonnegative")
        if self.redo and (self.dry_run or self.ids):
            raise ValueError("Redo cannot be combined with dry run or IDs")
        if self.note and not self.redo:
            raise ValueError("Note requires redo")


@dataclass(frozen=True)
class Result:
    resource_id: UUID
    status: str
    detail: str = ""


@dataclass
class Summary:
    results: list[Result] = field(default_factory=list)
    dispatched: int = 0

    @property
    def exit_code(self) -> int:
        return int(any(r.status in ("failed", "generation_uncertain", "lease_lost")
                       for r in self.results))


class Orchestrator:
    def __init__(self, repository: Repository, storage: Storage, describer: Describer,
                 generator: Generator, uploader: Uploader, settings: dict[str, Any],
                 lease_seconds: int = 300, secrets: tuple[str, ...] = ()):
        self.repository, self.storage = repository, storage
        self.describer, self.generator, self.uploader = describer, generator, uploader
        self.settings, self.lease_seconds, self.secrets = settings, lease_seconds, secrets

    def run(self, options: Options) -> Summary:
        summary = Summary()
        owner = str(uuid4())
        if not options.limit:
            return summary
        if options.redo:
            if not options.max_images:
                summary.results.append(Result(options.redo, "skipped", "Image cap; no writes"))
                return summary
            try:
                work = [self.repository.redo(options.redo, owner, self.lease_seconds, options.note)]
            except ClaimConflict:
                summary.results.append(Result(options.redo, "failed", "Redo refused: valid lease"))
                return summary
        else:
            work = self.repository.candidates(options.limit, options.ids, options.retry_failures)
        for item in work:
            if options.dry_run:
                try:
                    if item.attempt:
                        detail = f"Would resume from {item.attempt.stage}"
                    else:
                        description = validate(self.describer.describe(item.recipe))
                        build_prompt(item.recipe, description)
                        detail = "Description validated; would generate and upload"
                    summary.results.append(Result(item.recipe.id, "dry_run", detail))
                except Exception as exc:
                    summary.results.append(Result(item.recipe.id, "failed", redact(str(exc), self.secrets)))
                continue
            stage = item.attempt.stage if item.attempt else Stage.SELECTED
            if stage == Stage.FAILED:
                stage = item.attempt.last_successful_stage
            needs_generation = stage in (Stage.SELECTED, Stage.DESCRIBED, Stage.PROMPT_BUILT)
            if needs_generation and summary.dispatched >= options.max_images:
                summary.results.append(Result(item.recipe.id, "skipped", "Image cap; no writes"))
                continue
            try:
                attempt = (item.attempt if options.redo else
                           self.repository.claim(item, owner, self.lease_seconds))
            except ClaimConflict:
                summary.results.append(Result(item.recipe.id, "skipped", "Claim conflict"))
                continue
            except Exception as exc:
                summary.results.append(Result(item.recipe.id, "failed", redact(str(exc), self.secrets)))
                continue
            self._process(item, attempt, summary)
        return summary

    def _process(self, work: Work, attempt: Attempt, summary: Summary) -> None:
        repository, storage = self.repository, self.storage
        token = attempt.lease_token

        def update(**changes: Any) -> None:
            nonlocal attempt
            heartbeat.check()
            attempt = repository.update(attempt.id, token, **changes)

        def advance(stage: Stage, **changes: Any) -> None:
            update(stage=stage, last_successful_stage=stage, **changes)

        try:
            with Heartbeat(repository, attempt, self.lease_seconds) as heartbeat:
                try:
                    if attempt.stage == Stage.SELECTED:
                        description = heartbeat.call(lambda: self.describer.describe(work.recipe))
                        advance(Stage.DESCRIBED, description=validate(description), settings=self.settings)
                    if attempt.stage == Stage.DESCRIBED:
                        advance(Stage.PROMPT_BUILT,
                                final_prompt=build_prompt(work.recipe, attempt.description),
                                prompt_version=PROMPT_VERSION, settings=attempt.settings or self.settings)
                    if attempt.stage == Stage.GENERATING:
                        if storage.exists(attempt.original_storage_key):
                            advance(Stage.GENERATED, original_key=attempt.original_storage_key)
                        else:
                            raise GenerationError("Interrupted generation has no saved original", True)
                    if attempt.stage == Stage.PROMPT_BUILT:
                        # update() commits before returning, before any billed dispatch.
                        update(stage=Stage.GENERATING)
                        def generate() -> bytes:
                            summary.dispatched += 1
                            return self.generator.generate(attempt.final_prompt, attempt.settings)

                        original = heartbeat.call(generate)
                        heartbeat.call(lambda: storage.put(attempt.original_storage_key, original))
                        advance(Stage.GENERATED, original_key=attempt.original_storage_key)
                    if attempt.stage == Stage.GENERATED:
                        if storage.exists(attempt.jpeg_storage_key):
                            jpeg = fingerprint(storage.get(attempt.jpeg_storage_key))
                            verify(jpeg.data, jpeg.sha256, jpeg.length)
                        else:
                            jpeg = heartbeat.call(lambda: process(storage.get(attempt.original_key)))
                            heartbeat.call(lambda: storage.put(attempt.jpeg_storage_key, jpeg.data))
                        advance(Stage.PROCESSED, jpeg_key=attempt.jpeg_storage_key,
                                jpeg_sha256=jpeg.sha256, jpeg_length=jpeg.length)
                    if attempt.stage in (Stage.PROCESSED, Stage.PRESIGNED):
                        data = heartbeat.call(lambda: storage.get(attempt.jpeg_key))
                        verify(data, attempt.jpeg_sha256, attempt.jpeg_length)
                        presign = heartbeat.call(lambda: self.uploader.presign(
                            work.recipe.id, f"{attempt.id}.jpg", len(data)))
                        advance(Stage.PRESIGNED)
                        heartbeat.call(lambda: self.uploader.upload(presign, data))
                        # The key belongs to THIS successful PUT, never an earlier presign.
                        advance(Stage.UPLOADED, image_key=presign.image_key)
                    if attempt.stage == Stage.UPLOADED:
                        advance(Stage.COMPLETED)
                    summary.results.append(Result(work.recipe.id, str(attempt.stage)))
                except LeaseLost:
                    raise
                except Exception as exc:
                    heartbeat.check()
                    uncertain = isinstance(exc, GenerationError) and exc.uncertain
                    changes: dict[str, Any] = {}
                    if attempt.stage == Stage.GENERATING and not isinstance(exc, GenerationError):
                        if storage.exists(attempt.original_storage_key):
                            changes.update(last_successful_stage=Stage.GENERATED,
                                           original_key=attempt.original_storage_key)
                        else:
                            uncertain = True
                    status = Stage.UNCERTAIN if uncertain else Stage.FAILED
                    detail = redact(str(exc), self.secrets)
                    update(stage=status, error_type=type(exc).__name__, error_message=detail, **changes)
                    summary.results.append(Result(work.recipe.id, str(status), detail))
        except LeaseLost:
            # Never retry a rejected write, including error or release writes.
            summary.results.append(Result(work.recipe.id, "lease_lost", "Ownership lost; stopped"))
            return
        except Exception as exc:
            # A database/storage outage may prevent even recording an error.
            summary.results.append(Result(work.recipe.id, "failed", redact(str(exc), self.secrets)))
            return
        try:
            repository.release(attempt.id, token)
        except Exception:
            # Never retry release: another worker may already own the attempt.
            pass
