import json
import re
from dataclasses import asdict
from typing import Protocol

from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import BaseModel, Field

from .network import REQUEST_TIMEOUT_SECONDS, Checkpoint, retry_call
from .pricing import build_usage, estimate_cost
from .prompts import load_prompt
from .state import Recipe
from .storage import Storage


class DishDescription(BaseModel):
    description: str = Field(min_length=20, max_length=2000)


class BatchItem(BaseModel):
    id: str
    description: str


class BatchDescriptions(BaseModel):
    items: list[BatchItem]


class Describer(Protocol):
    def describe(self, recipe: Recipe, *, checkpoint: Checkpoint | None = None) -> str: ...


def validate_description(description: str) -> str:
    description = DishDescription(description=description.strip()).description
    if re.search(r"\b(sorry|apologi[sz]e|apology|cannot|can't|unable to|refuse|refusal)\b",
                 description, re.IGNORECASE):
        raise ValueError("Description contains refusal or apology text")
    return description


# Bump whenever prompt_templates/description_*.txt changes (tests/test_prompt_versions.py enforces it).
DESCRIPTION_VERSION = "tradition-aware-v2"

INSTRUCTIONS = load_prompt("description_instructions.txt", flatten=False)

# Marketing words the instructions forbid; enforced, not just requested.
STYLE_BLOCKLIST = re.compile(
    r"\b(perfect(?:ly)?|beautiful(?:ly)?|glisten\w*|gloss\w*|vibrant|succulent|juic\w*|tender|mouth-?watering|"
    r"delicious|tasty|savou?ry|flavou?r\w*|aroma\w*|comfort\w*|inviting|appeal\w*|elegant\w*|harmon\w*|"
    r"appetis\w*|appetiz\w*|delectable|indulgen\w*|rich|hearty|fresh|steaming|rustic|lush|decadent|gourmet)\b",
    re.IGNORECASE)


# The prompt asks for 300-650 characters; code enforces the hard limits so a pathological answer is repaired or rejected.
MIN_DESCRIPTION_CHARS = 200
MAX_DESCRIPTION_CHARS = 700


def own_words(recipe: Recipe) -> frozenset[str]:
    """Words from the recipe's own name and ingredient names ("Fresh Basil"): naming them is not marketing."""
    names = [recipe.name, *(str(i.get("name", "")) for i in recipe.ingredients)]
    return frozenset(re.findall(r"[a-z]+", " ".join(names).lower()))


def style_problems(description: str, allowed: frozenset[str] = frozenset()) -> list[str]:
    """Everything wrong with a description that the model should fix: banned words, then length.

    `allowed` holds words the recipe itself uses, which are never reported."""
    found = {m.group(0).lower() for m in STYLE_BLOCKLIST.finditer(description)}
    problems = sorted(found - allowed)
    if len(description) > MAX_DESCRIPTION_CHARS:
        problems.append(f"over {MAX_DESCRIPTION_CHARS} characters (shorten it)")
    elif len(description) < MIN_DESCRIPTION_CHARS:
        problems.append(f"under {MIN_DESCRIPTION_CHARS} characters (add visible detail)")
    return problems


def _sum_usage(first: dict | None, second: dict | None) -> dict | None:
    if not first or not second:
        return first or second
    merged = dict(first)
    for key in ("input_tokens", "output_text_tokens", "output_image_tokens"):
        merged[key] = int(first.get(key) or 0) + int(second.get(key) or 0)
    merged["cost_usd"] = estimate_cost(merged)
    return merged


DEFAULT_BATCH_SIZE = 10  # recipes per text request; keeps a batch well inside a 10k tokens-per-minute limit

BATCH_SUFFIX = load_prompt("description_batch_suffix.txt")


def _recipe_block(recipe: Recipe) -> str:
    """The recipe as untrusted data, fenced the way the instructions describe it."""
    return f"<recipe>\n{json.dumps(asdict(recipe), default=str)}\n</recipe>"


def _even_share(usage: dict | None, parts: int) -> dict | None:
    """One recipe's even share of a batched call, so per-recipe costs add up to the batch total."""
    if not usage or not parts:
        return None
    share = dict(usage)
    for key in ("input_tokens", "output_text_tokens", "output_image_tokens"):
        share[key] = int(usage.get(key) or 0) // parts
    share["cost_usd"] = estimate_cost(share)
    return share


class OpenAIDescriber:
    def __init__(self, client: OpenAI, model: str):
        self.client = client.with_options(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
        self.model = model
        self.last_usage: dict | None = None

    def _request(self, instructions: str, payload: str, text_format: type[BaseModel],
                 checkpoint: Checkpoint | None):
        # GPT-5 reasoning models do not accept a temperature setting.
        sampling = {} if self.model.startswith("gpt-5") else {"temperature": 0.2}
        response = retry_call(
            lambda: self.client.responses.parse(
                model=self.model, instructions=instructions, input=payload, text_format=text_format, **sampling,
            ),
            # 429 is deliberately not retried: an immediate retry is also rate limited and consumes more quota.
            lambda exc: isinstance(exc, APIConnectionError) or (
                isinstance(exc, APIStatusError) and (exc.status_code == 408 or exc.status_code >= 500)),
            checkpoint=checkpoint,
        )
        usage = getattr(response, "usage", None)
        self.last_usage = _sum_usage(self.last_usage, build_usage(
            "openai", self.model, input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_text_tokens=getattr(usage, "output_tokens", 0) or 0) if usage else None)
        if response.output_parsed is None:
            raise ValueError("Description refused or missing structured output")
        return response.output_parsed

    def _call(self, recipe: Recipe, checkpoint: Checkpoint | None, *, previous: str = "", problems: list[str] = ()):
        instructions, payload = INSTRUCTIONS, _recipe_block(recipe)
        if previous:
            # Editing a given text fixes banned words far more reliably than asking for a fresh one.
            instructions += (" Your previous description is in <previous_description>. It has these problems: "
                             + ", ".join(problems) + ". Rewrite it with the same content, replacing each forbidden word "
                             "with plain visible detail or removing it, and fixing any length problem.")
            payload += f"\n<previous_description>\n{previous}\n</previous_description>"
        parsed = self._request(instructions, payload, DishDescription, checkpoint)
        return validate_description(parsed.description)

    def describe(self, recipe: Recipe, *, checkpoint: Checkpoint | None = None) -> str:
        self.last_usage = None
        description = self._call(recipe, checkpoint)
        problems = style_problems(description, own_words(recipe))
        if problems:
            # One bounded repair pass; both calls are counted in last_usage. Still-bad text fails the recipe.
            description = self._call(recipe, checkpoint, previous=description, problems=problems)
            problems = style_problems(description, own_words(recipe))
            if problems:
                raise ValueError("Description still has style problems: " + ", ".join(problems))
        return description

    def describe_batch(self, recipes: list[Recipe], *,
                       checkpoint: Checkpoint | None = None) -> dict[str, str]:
        """One request for many recipes. Returns {recipe id: description} for the valid, style-clean results only."""
        self.last_usage = None
        parsed = self._request(f"{INSTRUCTIONS}\n\n{BATCH_SUFFIX}",
                               "\n".join(_recipe_block(r) for r in recipes),
                               BatchDescriptions, checkpoint)
        by_id = {str(r.id): r for r in recipes}
        wanted = set(by_id)
        found: dict[str, str] = {}
        for item in parsed.items:
            if item.id not in wanted or item.id in found:
                continue
            try:
                text = validate_description(item.description)
            except ValueError:
                continue
            if not style_problems(text, own_words(by_id[item.id])):
                found[item.id] = text
        return found


class BatchedDescriber:
    """Describes many recipes per request. Each result is saved to storage the moment it arrives, so a rerun
    never repeats a paid request; any recipe the batch missed falls back to a single-recipe call."""

    def __init__(self, inner: OpenAIDescriber, storage: Storage, batch_size: int = DEFAULT_BATCH_SIZE):
        self.inner, self.storage, self.batch_size = inner, storage, batch_size
        self.last_usage: dict | None = None
        self._ready: dict[str, tuple[str, dict | None]] = {}  # recipe id -> (description, usage to bill)

    def _key(self, recipe: Recipe) -> str:
        return f"descriptions/{DESCRIPTION_VERSION}-{self.inner.model}/{recipe.id}.json"

    def prefetch(self, recipes: list[Recipe], *, persist: bool = True) -> None:
        missing = []
        for recipe in recipes:
            if self.storage.exists(self._key(recipe)):  # paid for in an earlier run: no request, no cost
                saved = json.loads(self.storage.get(self._key(recipe)))
                self._ready[str(recipe.id)] = (saved["description"], None)
            else:
                missing.append(recipe)
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start:start + self.batch_size]
            found = self.inner.describe_batch(chunk)
            usage = _even_share(self.inner.last_usage, len(found))
            for recipe in chunk:
                if str(recipe.id) in found:
                    if persist:
                        self.storage.put(self._key(recipe),
                                         json.dumps({"description": found[str(recipe.id)]}).encode())
                    self._ready[str(recipe.id)] = (found[str(recipe.id)], usage)

    def describe(self, recipe: Recipe, *, checkpoint: Checkpoint | None = None) -> str:
        ready = self._ready.pop(str(recipe.id), None)
        if ready is None:
            description = self.inner.describe(recipe, checkpoint=checkpoint)
            self.last_usage = self.inner.last_usage
            return description
        description, self.last_usage = ready
        return description
