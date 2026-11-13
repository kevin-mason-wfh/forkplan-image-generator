from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from fakes import plain
from openai import APIStatusError

from forkplan_images.descriptions import (
    BatchDescriptions,
    BatchedDescriber,
    BatchItem,
    DishDescription,
    OpenAIDescriber,
    validate_description,
)
from forkplan_images.prompts import PROMPT_VERSION, build_prompt
from forkplan_images.state import LeaseLost, Recipe
from forkplan_images.storage import LocalStorage


@pytest.mark.parametrize("value", ["short", "x" * 2001,
                                   "I am sorry, I cannot describe this recipe.",
                                   "An apology for being unable to produce this."])
def test_description_rejection(value):
    with pytest.raises(ValueError):
        validate_description(value)


def test_typed_parsing_and_no_ingredient_name_requirement():
    client = Mock()
    client.with_options.return_value = client
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=DishDescription(description=plain("A smooth golden soup in a ceramic bowl.")))
    recipe = Recipe(uuid4(), "Soup", ingredients=({"name": "Onion", "quantity": 2, "unit": ""},))
    description = OpenAIDescriber(client, "configured-model").describe(recipe)
    arguments = client.responses.parse.call_args.kwargs
    assert arguments["text_format"] == DishDescription
    assert "Show only the listed ingredients" in arguments["instructions"]
    assert "Onion" in arguments["input"]
    assert "Onion" not in description
    prompt = build_prompt(recipe, description)
    assert "white-washed" in prompt and "wooden table" in prompt
    assert "Quantities (cooked): 2 Onion" in prompt
    assert PROMPT_VERSION
    client.responses.parse.return_value = SimpleNamespace(output_parsed=None)
    with pytest.raises(ValueError):
        OpenAIDescriber(client, "configured-model").describe(recipe)


def test_text_retries_are_fenced_on_ownership_loss():
    client = Mock()
    client.with_options.return_value = client
    response = httpx.Response(503, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    client.responses.parse.side_effect = APIStatusError("unavailable", response=response, body={})
    checkpoint = Mock(side_effect=[None, LeaseLost("Ownership lost")])
    with pytest.raises(LeaseLost):
        OpenAIDescriber(client, "configured-model").describe(
            Recipe(uuid4(), "Soup"), checkpoint=checkpoint)
    assert client.responses.parse.call_count == 1
    client.with_options.assert_called_once_with(timeout=60.0, max_retries=0)


def test_prompt_omits_null_quantity_and_unit():
    from uuid import uuid4

    from forkplan_images.prompts import build_prompt
    from forkplan_images.state import Recipe
    recipe = Recipe(id=uuid4(), name="Prawns", ingredients=(
        {"name": "Lemon", "quantity": None, "unit": None},
        {"name": "Butter", "quantity": 28, "unit": "g"},
        {"name": "Salt", "quantity": None, "unit": "pinch"},
    ))
    prompt = build_prompt(recipe, "A plated dish of prawns in butter.")
    assert "None" not in prompt
    assert "Quantities (cooked): Lemon; 28 g Butter; pinch Salt" in prompt


def _client(*descriptions):
    client = Mock()
    client.with_options.return_value = client
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)
    client.responses.parse.side_effect = [
        SimpleNamespace(output_parsed=DishDescription(description=d), usage=usage) for d in descriptions]
    return client


def test_marketing_words_trigger_one_repair_and_both_calls_are_costed():
    client = _client(plain("A perfectly seared steak with glistening, creamy mash on a plate."),
                     plain("A seared steak beside a mound of cauliflower mash on a plate."))
    describer = OpenAIDescriber(client, "gpt-4o-2024-08-06")
    text = describer.describe(Recipe(uuid4(), "Steak"))
    assert text.startswith("A seared steak") and client.responses.parse.call_count == 2
    assert "glistening, perfectly" in client.responses.parse.call_args.kwargs["instructions"]
    assert "A perfectly seared steak" in client.responses.parse.call_args.kwargs["input"]  # edits the previous text
    assert describer.last_usage["input_tokens"] == 200 and describer.last_usage["output_text_tokens"] == 40


def test_persistent_marketing_language_fails_the_recipe_after_one_repair():
    client = _client(plain("A perfectly seared steak with mash on a plate, nicely done."),
                     plain("A succulent steak with mash on a plate, nicely done."))
    with pytest.raises(ValueError, match="style problems"):
        OpenAIDescriber(client, "gpt-4o-2024-08-06").describe(Recipe(uuid4(), "Steak"))
    assert client.responses.parse.call_count == 2


# --- batched descriptions ---------------------------------------------------



def _batch_client(*batches):
    """Each element is a list of (recipe, text); one request is answered per element."""
    client = Mock()
    client.with_options.return_value = client
    usage = SimpleNamespace(input_tokens=600, output_tokens=300)
    client.responses.parse.side_effect = [
        SimpleNamespace(output_parsed=BatchDescriptions(
            items=[BatchItem(id=str(r.id), description=t) for r, t in batch]), usage=usage)
        for batch in batches]
    return client


def _recipes(count):
    return [Recipe(uuid4(), f"Dish {i}") for i in range(count)]


def test_batch_uses_one_request_and_splits_cost_across_recipes(tmp_path):
    recipes = _recipes(3)
    client = _batch_client([(r, plain(f"A plain plate holding dish number {i} as cooked.")) for i, r in enumerate(recipes)])
    describer = BatchedDescriber(OpenAIDescriber(client, "gpt-4o-2024-08-06"), LocalStorage(tmp_path))
    describer.prefetch(recipes)
    assert client.responses.parse.call_count == 1
    assert "<recipe> blocks" in client.responses.parse.call_args.kwargs["instructions"]
    assert client.responses.parse.call_args.kwargs["input"].count("<recipe>") == 3
    assert "dish number 1 as cooked." in describer.describe(recipes[1])
    assert describer.last_usage["input_tokens"] == 200 and describer.last_usage["output_text_tokens"] == 100
    assert client.responses.parse.call_count == 1  # describe() made no further request


def test_batch_results_are_saved_so_a_rerun_makes_no_request(tmp_path):
    recipes = _recipes(2)
    client = _batch_client([(r, plain("A plain plate holding the cooked dish.")) for r in recipes])
    BatchedDescriber(OpenAIDescriber(client, "gpt-4o-2024-08-06"), LocalStorage(tmp_path)).prefetch(recipes)
    rerun_client = _batch_client()
    rerun = BatchedDescriber(OpenAIDescriber(rerun_client, "gpt-4o-2024-08-06"), LocalStorage(tmp_path))
    rerun.prefetch(recipes)
    assert rerun.describe(recipes[0]) == plain("A plain plate holding the cooked dish.")
    assert rerun.last_usage is None  # already paid for in the first run
    rerun_client.responses.parse.assert_not_called()


def test_dry_run_prefetch_saves_nothing(tmp_path):
    recipes = _recipes(1)
    client = _batch_client([(recipes[0], plain("A plain plate holding the cooked dish."))])
    BatchedDescriber(OpenAIDescriber(client, "m"), LocalStorage(tmp_path)).prefetch(recipes, persist=False)
    assert not list(tmp_path.rglob("*.json"))


def test_chunks_of_batch_size(tmp_path):
    recipes = _recipes(5)
    client = _batch_client(*[[(r, plain("A plain plate holding the cooked dish.")) for r in recipes[i:i + 2]]
                             for i in range(0, 5, 2)])
    BatchedDescriber(OpenAIDescriber(client, "m"), LocalStorage(tmp_path), batch_size=2).prefetch(recipes)
    assert client.responses.parse.call_count == 3


def test_missing_invalid_and_marketing_results_fall_back_to_single_calls(tmp_path):
    ok, missing, marketing = _recipes(3)
    client = _batch_client([(ok, plain("A plain plate holding the cooked dish.")),
                            (marketing, plain("A perfectly seared dish on a plate, nicely done."))])
    describer = BatchedDescriber(OpenAIDescriber(client, "m"), LocalStorage(tmp_path))
    describer.prefetch([ok, missing, marketing])
    client.responses.parse.side_effect = [
        SimpleNamespace(output_parsed=DishDescription(description=plain("A plain dish on a plate, as cooked.")), usage=None)]
    assert describer.describe(ok) == plain("A plain plate holding the cooked dish.")
    assert describer.describe(missing) == plain("A plain dish on a plate, as cooked.")
    assert client.responses.parse.call_count == 2  # one batch + one single fallback


def test_rate_limit_is_not_retried_because_a_retry_also_spends_quota():
    client = Mock()
    client.with_options.return_value = client
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    client.responses.parse.side_effect = APIStatusError("rate limited", response=response, body={})
    with pytest.raises(APIStatusError):
        OpenAIDescriber(client, "configured-model").describe(Recipe(uuid4(), "Soup"))
    assert client.responses.parse.call_count == 1


def test_length_limits_are_enforced_with_one_repair():
    from forkplan_images.descriptions import MAX_DESCRIPTION_CHARS, style_problems
    too_long = "A plain plate holding the cooked dish. " * 30
    assert len(too_long) > MAX_DESCRIPTION_CHARS and "over 700" in style_problems(too_long)[0]
    assert "under 200" in style_problems("A plain plate.")[0]
    client = _client(too_long[:1500], plain("A plain plate holding the cooked dish."))
    text = OpenAIDescriber(client, "gpt-4o-2024-08-06").describe(Recipe(uuid4(), "Soup"))
    assert client.responses.parse.call_count == 2 and "over 700 characters" in client.responses.parse.call_args.kwargs["instructions"]
    assert 200 <= len(text) <= 700


def test_added_mood_words_are_flagged():
    from forkplan_images.descriptions import style_problems
    assert style_problems(plain("A rich, hearty and rustic stew, steaming and fresh.")) == ["fresh", "hearty", "rich", "rustic", "steaming"]


def test_words_from_the_recipes_own_ingredient_names_are_not_flagged():
    from forkplan_images.descriptions import own_words, style_problems
    recipe = Recipe(uuid4(), "Margherita Pizza", ingredients=({"name": "Fresh Basil"}, {"name": "Mozzarella"}))
    text = plain("A round pizza topped with fresh basil leaves and rich, hearty cheese.")
    assert style_problems(text) == ["fresh", "hearty", "rich"]
    assert style_problems(text, own_words(recipe)) == ["hearty", "rich"]
