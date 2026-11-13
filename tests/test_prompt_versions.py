"""A prompt change must come with a version bump: cached descriptions and stored attempts are keyed by version."""
from hashlib import sha256
from uuid import UUID

from forkplan_images.descriptions import BATCH_SUFFIX, DESCRIPTION_VERSION, INSTRUCTIONS
from forkplan_images.prompts import PROMPT_VERSION, build_prompt
from forkplan_images.state import Recipe

# version -> sha256 of the prompt text. When a prompt changes, bump its version and add the new hash here.
PINNED = {
    PROMPT_VERSION: "1c5c67e75e18651cbc8cd42783d1f7433a49653a47ce02d6ef26175ce2fa7e4b",
    DESCRIPTION_VERSION: "ef263ae5c1b83e769ea101c85e01a054ba873bd02d166c97131f2a4b01503471",
}


def digest(*texts: str) -> str:
    return sha256("\n".join(texts).encode()).hexdigest()


SAMPLE = Recipe(UUID(int=1), "Sample Dish", instructions="Cook 2 Eggs.",
                ingredients=({"name": "Eggs", "quantity": 2, "unit": "each"},))


def test_image_prompt_text_is_pinned_to_its_version():
    # The whole built prompt (name line, recipe layout and style file), not just the style file.
    assert digest(build_prompt(SAMPLE, "A plain description of the dish.")) == PINNED[PROMPT_VERSION]


def test_description_prompt_text_is_pinned_to_its_version():
    assert digest(INSTRUCTIONS, BATCH_SUFFIX) == PINNED[DESCRIPTION_VERSION]
