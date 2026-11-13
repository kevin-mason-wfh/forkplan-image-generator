"""Token usage capture and cost estimates.

Token counts come straight from the provider responses and are authoritative. Rates are USD per
1M tokens and MUST be confirmed against the provider pricing pages (they change); a model with no
known rate is recorded with cost_usd = None rather than a guess.
"""
import csv
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

# rates: input (text/image in), output_text (incl. reasoning), output_image
RATES: dict[str, dict[str, float]] = {
    "gpt-4o-2024-08-06": {"input": 2.50, "output_text": 10.00},
    # Image rate inferred from 1,120 tokens = $0.0672 per 1K image; verify at ai.google.dev/pricing.
    # UNVERIFIED third-party aggregator figures ($5 in / $30 image out per 1M); aggregators disagree, so confirm
    # on the OpenAI pricing page. Image *input* tokens may be billed higher ($8) than text input.
    "gpt-image-2": {"input": 5.00, "output_image": 30.00},
    "gemini-3.1-flash-image": {"input": 0.50, "output_text": 3.00, "output_image": 60.00},
    # UNVERIFIED third-party figures (~$0.134 and ~$0.034 per 1K image); confirm at ai.google.dev/pricing.
    "gemini-3-pro-image": {"input": 2.00, "output_text": 12.00, "output_image": 120.00},
    "gemini-3.1-flash-lite-image": {"input": 0.25, "output_text": 1.50, "output_image": 30.00},
}


def build_usage(provider: str, model: str, *, input_tokens: int = 0, output_text_tokens: int = 0,
                output_image_tokens: int = 0) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "provider": provider, "model": model, "input_tokens": int(input_tokens),
        "output_text_tokens": int(output_text_tokens), "output_image_tokens": int(output_image_tokens),
    }
    usage["cost_usd"] = estimate_cost(usage)
    return usage


def estimate_cost(usage: Mapping[str, Any]) -> float | None:
    rates = RATES.get(str(usage.get("model")))
    if not rates:
        return None
    parts = (("input_tokens", "input"), ("output_text_tokens", "output_text"),
             ("output_image_tokens", "output_image"))
    total = 0.0
    for tokens_key, rate_key in parts:
        tokens = int(usage.get(tokens_key) or 0)
        if tokens and rate_key not in rates:
            return None  # unpriced token class: do not under-report
        total += tokens * rates.get(rate_key, 0.0) / 1_000_000
    return round(total, 6)


def total_cost(*usages: Mapping[str, Any] | None) -> tuple[float, int]:
    """Return (known cost, number of usage records without a price)."""
    known, unpriced = 0.0, 0
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        cost = usage.get("cost_usd")
        if cost is None:
            unpriced += 1
        else:
            known += float(cost)
    return known, unpriced


LEDGER_COLUMNS = (
    "timestamp_utc", "run_kind", "recipe_id", "recipe_name", "attempt_id", "attempt_number", "status",
    "prompt_version", "text_model", "text_input_tokens", "text_output_tokens", "text_cost_usd",
    "image_model", "image_input_tokens", "image_output_text_tokens", "image_output_image_tokens",
    "image_cost_usd", "total_cost_usd", "unpriced_calls",
)


class CostLedger:
    """Append-only local CSV: one row per recipe per run that made a billable call (also dry runs)."""

    def __init__(self, path: Path):
        self.path, self._lock = Path(path), Lock()

    def record(self, *, run_kind: str, recipe_id: object, recipe_name: str, attempt_id: object = "",
               attempt_number: object = "", status: str, prompt_version: str = "",
               text: Mapping[str, Any] | None = None, image: Mapping[str, Any] | None = None) -> None:
        text, image = text or {}, image or {}
        known, unpriced = total_cost(text or None, image or None)
        row = {
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"), "run_kind": run_kind,
            "recipe_id": recipe_id, "recipe_name": recipe_name, "attempt_id": attempt_id,
            "attempt_number": attempt_number, "status": status, "prompt_version": prompt_version,
            "text_model": text.get("model", ""), "text_input_tokens": text.get("input_tokens", ""),
            "text_output_tokens": text.get("output_text_tokens", ""), "text_cost_usd": text.get("cost_usd", ""),
            "image_model": image.get("model", ""), "image_input_tokens": image.get("input_tokens", ""),
            "image_output_text_tokens": image.get("output_text_tokens", ""),
            "image_output_image_tokens": image.get("output_image_tokens", ""),
            "image_cost_usd": image.get("cost_usd", ""), "total_cost_usd": round(known, 6),
            "unpriced_calls": unpriced,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.exists()
            with open(self.path, "a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
                if new:
                    writer.writeheader()
                writer.writerow(row)
