"""Runs one claimed attempt through the stage machine, one small method per stage."""
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .descriptions import DESCRIPTION_VERSION, Describer, validate_description
from .generation import GenerationError, Generator
from .heartbeat import Heartbeat
from .imaging import fingerprint, process, verify
from .network import auth_failure_status
from .prompts import PROMPT_VERSION, build_prompt
from .redaction import redact
from .repository import Repository, Work
from .state import Attempt, LeaseLost, Recipe, Stage
from .storage import Storage
from .uploads import Uploader

log = logging.getLogger(__name__)

ExportCopy = Callable[[Recipe, Attempt, bytes], None]  # saves a verified JPEG somewhere readable


@dataclass(frozen=True)
class Services:
    """Everything an attempt talks to. Built per attempt so swapped-in collaborators are always honoured."""
    repository: Repository
    storage: Storage
    describer: Describer
    generator: Generator
    uploader: Uploader
    settings: dict[str, Any]
    secrets: tuple[str, ...]
    export_copy: ExportCopy | None = None


@dataclass(frozen=True)
class Outcome:
    status: str
    detail: str = ""
    stop_run: bool = False  # the credentials were rejected: continuing would only repeat the failure


class AttemptRunner:
    """Advances an attempt from its current stage to completion, or records why it stopped.

    Every stage method first checks the attempt is at its stage, so a resumed attempt simply skips what is
    already done. A rejected lease write (LeaseLost) is never caught here: the caller must stop immediately.
    """

    def __init__(self, services: Services, work: Work, attempt: Attempt, heartbeat: Heartbeat,
                 on_image_dispatch: Callable[[], None]):
        self.services, self.work, self.attempt, self.heartbeat = services, work, attempt, heartbeat
        self.on_image_dispatch = on_image_dispatch
        self.label = f"'{work.recipe.name}' ({str(work.recipe.id)[:8]})"
        self.usage: dict[str, dict[str, Any]] = {}  # {'text': ..., 'image': ...}: calls made in THIS run only

    def run(self) -> Outcome:
        try:
            for stage_step in (self._describe, self._build_prompt, self._recover_interrupted_generation,
                               self._generate, self._process_image, self._publish, self._complete):
                stage_step()
            return Outcome(str(self.attempt.stage))
        except LeaseLost:
            raise
        except Exception as exc:
            return self._record_failure(exc)

    # -- persistence helpers -------------------------------------------------------------------------

    def _update(self, **changes: Any) -> None:
        self.heartbeat.check()
        self.attempt = self.services.repository.update(self.attempt.id, self.attempt.lease_token, **changes)

    def _advance(self, stage: Stage, **changes: Any) -> None:
        self._update(stage=stage, last_successful_stage=stage, **changes)
        log.info("%s: %s", self.label, stage.value.replace("_", " "))

    def _capture_usage(self, kind: str, provider: object) -> dict[str, Any]:
        """Remember the provider's last call usage for cost reporting, and return it as columns to persist."""
        usage = getattr(provider, "last_usage", None)
        if not isinstance(usage, dict):
            return {}
        self.usage[kind] = usage
        return {f"{kind}_usage": usage}

    # -- stages --------------------------------------------------------------------------------------

    def _describe(self) -> None:
        if self.attempt.stage != Stage.SELECTED:
            return
        services = self.services
        if self.attempt.description and self.attempt.description_version == DESCRIPTION_VERSION:
            # A retry or redo carries the earlier description: no text call, so no text_usage on this attempt.
            log.info("%s: reusing the saved description (no text call)", self.label)
            self._advance(Stage.DESCRIBED, settings=services.settings)
            return
        try:
            description = self.heartbeat.call(lambda: services.describer.describe(
                self.work.recipe, checkpoint=self.heartbeat.fence))
        finally:
            usage_columns = self._capture_usage("text", services.describer)  # billed even if the text is rejected
        self._advance(Stage.DESCRIBED, description=validate_description(description),
                      description_version=DESCRIPTION_VERSION, settings=services.settings, **usage_columns)

    def _build_prompt(self) -> None:
        if self.attempt.stage != Stage.DESCRIBED:
            return
        self._advance(Stage.PROMPT_BUILT, final_prompt=build_prompt(self.work.recipe, self.attempt.description),
                      prompt_version=PROMPT_VERSION, settings=self.attempt.settings or self.services.settings)

    def _recover_interrupted_generation(self) -> None:
        """A crash after the image call was committed: use the saved original, or give up (never regenerate)."""
        if self.attempt.stage != Stage.GENERATING:
            return
        if not self.services.storage.exists(self.attempt.original_storage_key):
            raise GenerationError("Interrupted generation has no saved original", True)
        self._advance(Stage.GENERATED)

    def _generate(self) -> None:
        if self.attempt.stage != Stage.PROMPT_BUILT:
            return
        services = self.services
        self._update(stage=Stage.GENERATING)  # committed before the billed dispatch below
        log.info("%s: requesting the image (can take a minute or two)", self.label)

        def dispatch() -> bytes:
            self.on_image_dispatch()
            return services.generator.generate(self.attempt.final_prompt, self.attempt.settings,
                                               checkpoint=self.heartbeat.fence)

        original = self.heartbeat.call(dispatch)
        usage_columns = self._capture_usage("image", services.generator)
        self.heartbeat.call(lambda: services.storage.put(self.attempt.original_storage_key, original))
        self._advance(Stage.GENERATED, **usage_columns)

    def _process_image(self) -> None:
        if self.attempt.stage != Stage.GENERATED:
            return
        storage, attempt = self.services.storage, self.attempt
        if storage.exists(attempt.jpeg_storage_key):
            jpeg = fingerprint(storage.get(attempt.jpeg_storage_key))
            verify(jpeg.data, jpeg.sha256, jpeg.length)
        else:
            jpeg = self.heartbeat.call(lambda: process(storage.get(attempt.original_storage_key)))
            self.heartbeat.call(lambda: storage.put(attempt.jpeg_storage_key, jpeg.data))
        self._advance(Stage.PROCESSED,
                      jpeg_sha256=jpeg.sha256, jpeg_length=jpeg.length)

    def _publish(self) -> None:
        if self.attempt.stage not in (Stage.PROCESSED, Stage.PRESIGNED):
            return
        services, attempt, recipe = self.services, self.attempt, self.work.recipe
        data = self.heartbeat.call(lambda: services.storage.get(attempt.jpeg_storage_key))
        verify(data, attempt.jpeg_sha256, attempt.jpeg_length)
        if services.export_copy:
            # Verified bytes only; idempotent, so resuming here never duplicates.
            self.heartbeat.call(lambda: services.export_copy(recipe, attempt, data))
        presign = self.heartbeat.call(lambda: services.uploader.presign(
            recipe.id, f"{attempt.id}.jpg", len(data), checkpoint=self.heartbeat.fence))
        self._advance(Stage.PRESIGNED)
        self.heartbeat.call(lambda: services.uploader.upload(presign, data, checkpoint=self.heartbeat.fence))
        # The key belongs to THIS successful PUT, never an earlier presign.
        self._advance(Stage.UPLOADED, image_key=presign.image_key)

    def _complete(self) -> None:
        if self.attempt.stage == Stage.UPLOADED:
            self._advance(Stage.COMPLETED)

    # -- failure -------------------------------------------------------------------------------------

    def _record_failure(self, exc: Exception) -> Outcome:
        self.heartbeat.check()
        storage = self.services.storage
        uncertain = isinstance(exc, GenerationError) and exc.uncertain
        changes: dict[str, Any] = {}
        if self.attempt.stage == Stage.GENERATING and not isinstance(exc, GenerationError):
            # Something other than the image call failed while the call was in flight: keep the original if saved.
            if storage.exists(self.attempt.original_storage_key):
                changes.update(last_successful_stage=Stage.GENERATED)
            else:
                uncertain = True
        status = Stage.UNCERTAIN if uncertain else Stage.FAILED
        detail = redact(str(exc), self.services.secrets)
        rejected = auth_failure_status(exc)
        if rejected:
            service = "OpenAI" if type(exc).__module__.startswith("openai") else "A service"
            detail = (f"{service} rejected the credentials (HTTP {rejected}). Check the API token or key in your "
                      "configuration. Saved work is kept: fix it, then run --retry-failures.")
        for kind, usage in self.usage.items():  # calls billed this run that no stage write has saved yet
            if not getattr(self.attempt, f"{kind}_usage"):
                changes[f"{kind}_usage"] = usage
        self._update(stage=status, error_type=type(exc).__name__, error_message=detail, **changes)
        log.warning("%s: attempt %s ended as %s: %s", self.label, self.attempt.attempt_number, status, detail)
        return Outcome(str(status), detail, stop_run=bool(rejected))
