import base64

import httpx
import pytest
import respx
from test_orchestrator import setup

from forkplan_images.generation import GEMINI_URL, GeminiGenerator
from forkplan_images.orchestrator import Options
from forkplan_images.pricing import build_usage, estimate_cost, total_cost
from forkplan_images.state import Stage


def test_pricing_known_and_unknown_models():
    text = build_usage("openai", "gpt-4o-2024-08-06", input_tokens=1000, output_text_tokens=200)
    assert text["cost_usd"] == pytest.approx(1000 * 2.5 / 1e6 + 200 * 10 / 1e6)
    image = build_usage("gemini", "gemini-3.1-flash-image", input_tokens=13,
                        output_text_tokens=392, output_image_tokens=1120)
    assert image["cost_usd"] == pytest.approx(0.0672 + 13 * 0.5 / 1e6 + 392 * 3 / 1e6, abs=1e-6)
    # Unknown model, or an unpriced token class, never reports a misleadingly low number.
    assert build_usage("openai", "gpt-image-9", output_image_tokens=1000)["cost_usd"] is None
    assert estimate_cost({"model": "gpt-4o-2024-08-06", "output_image_tokens": 5}) is None
    assert total_cost(text, {"cost_usd": None}, None, "junk") == (text["cost_usd"], 1)


@respx.mock
def test_gemini_usage_metadata_is_captured():
    model = {"image_model": "gemini-3.1-flash-image"}
    body = {"candidates": [{"content": {"parts": [{"inlineData": {
        "data": base64.b64encode(b"img").decode()}}]}}],
        "usageMetadata": {"promptTokenCount": 13, "candidatesTokenCount": 1512,
                          "candidatesTokensDetails": [{"modality": "IMAGE", "tokenCount": 1120}]}}
    respx.post(GEMINI_URL.format(model=model["image_model"])).respond(200, json=body)
    with httpx.Client() as client:
        generator = GeminiGenerator(client, "k")
        generator.generate("prompt", model)
    usage = generator.last_usage
    assert (usage["input_tokens"], usage["output_image_tokens"], usage["output_text_tokens"]) == (13, 1120, 392)
    assert usage["cost_usd"] == pytest.approx(0.0672 + 13 * 0.5 / 1e6 + 392 * 3 / 1e6, abs=1e-6)


def test_orchestrator_records_usage_and_run_cost(tmp_path):
    runner, repo, store, describe, generate, upload, recipes = setup(tmp_path)
    describe.last_usage = build_usage("openai", "gpt-4o-2024-08-06", input_tokens=1000, output_text_tokens=100)
    generate.last_usage = build_usage("gemini", "gemini-3.1-flash-image", output_image_tokens=1120)
    summary = runner.run(Options())
    attempt = next(iter(repo.attempts.values()))
    assert attempt.stage == Stage.COMPLETED
    assert attempt.text_usage["input_tokens"] == 1000 and attempt.image_usage["output_image_tokens"] == 1120
    expected = describe.last_usage["cost_usd"] + generate.last_usage["cost_usd"]
    assert summary.cost_usd == pytest.approx(expected) and summary.unpriced_calls == 0
    assert summary.results[0].cost_usd == pytest.approx(expected, abs=1e-6)


def test_orchestrator_ignores_missing_usage(tmp_path):
    runner, repo, *_ = setup(tmp_path)  # Mock fakes expose non-dict last_usage
    summary = runner.run(Options())
    attempt = next(iter(repo.attempts.values()))
    assert attempt.text_usage is None and attempt.image_usage is None
    assert summary.cost_usd == 0 and summary.results[0].cost_usd is None


def test_cost_ledger_rows_for_generate_and_dry_run(tmp_path):
    import csv

    from forkplan_images.pricing import LEDGER_COLUMNS, CostLedger
    runner, repo, store, describe, generate, upload, recipes = setup(tmp_path)
    runner.ledger = CostLedger(tmp_path / "logs" / "cost_log.csv")
    describe.last_usage = build_usage("openai", "gpt-4o-2024-08-06", input_tokens=1000, output_text_tokens=100)
    generate.last_usage = build_usage("gemini", "gemini-3.1-flash-image", output_image_tokens=1120)
    runner.run(Options(dry_run=True))                     # dry run: text call only, still logged
    runner.run(Options())                                 # real run: text + image
    rows = list(csv.DictReader((tmp_path / "logs" / "cost_log.csv").read_text().splitlines()))
    assert list(rows[0]) == list(LEDGER_COLUMNS) and [r["run_kind"] for r in rows] == ["dry_run", "generate"]
    assert rows[0]["image_model"] == "" and float(rows[0]["total_cost_usd"]) == pytest.approx(
        describe.last_usage["cost_usd"], abs=1e-6)
    assert rows[1]["image_model"] == "gemini-3.1-flash-image" and rows[1]["status"] == "completed"
    assert float(rows[1]["total_cost_usd"]) == pytest.approx(
        describe.last_usage["cost_usd"] + generate.last_usage["cost_usd"], abs=1e-6)
