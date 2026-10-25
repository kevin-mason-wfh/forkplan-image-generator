from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import httpx
from openai import APIStatusError

from forkplan_images.descriptions import DishDescription, OpenAIDescriber, validate
from forkplan_images.prompts import PROMPT_VERSION, build_prompt
from forkplan_images.state import Recipe
from forkplan_images.state import LeaseLost


@pytest.mark.parametrize("value", ["short", "x" * 2001,
                                   "I am sorry, I cannot describe this recipe.",
                                   "An apology for being unable to produce this."])
def test_description_rejection(value):
    with pytest.raises(ValueError):
        validate(value)


def test_typed_parsing_and_no_ingredient_name_requirement():
    client = Mock()
    client.with_options.return_value = client
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=DishDescription(description="A smooth golden soup in a ceramic bowl."))
    recipe = Recipe(uuid4(), "Soup", ingredients=({"name": "Onion", "quantity": 2, "unit": ""},))
    description = OpenAIDescriber(client, "configured-model").describe(recipe)
    arguments = client.responses.parse.call_args.kwargs
    assert arguments["text_format"] == DishDescription
    assert "Invent no ingredients" in arguments["instructions"]
    assert "Onion" in arguments["input"]
    assert "Onion" not in description
    prompt = build_prompt(recipe, description)
    assert "white wooden table" in prompt
    assert "2  Onion" in prompt
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
