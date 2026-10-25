from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from PIL import Image
import pytest

from forkplan_images.imaging import process, verify
from forkplan_images.storage import LocalStorage


def png():
    output = BytesIO()
    Image.new("RGBA", (256, 256), (255, 0, 0, 128)).save(output, format="PNG")
    return output.getvalue()


def test_original_immutable_and_atomic(tmp_path):
    store = LocalStorage(tmp_path)
    store.put("attempts/a/original", png())
    with pytest.raises(FileExistsError):
        store.put("attempts/a/original", b"replacement")
    assert store.get("attempts/a/original") == png()
    assert not list(tmp_path.rglob(".pending-*"))
    for key in ("/absolute", "../escape", "attempts/../../escape"):
        with pytest.raises(ValueError):
            store.put(key, b"x")


def test_concurrent_writes_only_one_wins(tmp_path):
    store = LocalStorage(tmp_path)

    def write(value):
        try:
            store.put("original", value)
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(2) as executor:
        assert sum(executor.map(write, [b"a" * 10000, b"b" * 10000])) == 1
    assert store.get("original") in (b"a" * 10000, b"b" * 10000)


def test_jpeg_size_and_integrity():
    jpeg = process(png())
    assert jpeg.data.startswith(b"\xff\xd8")
    assert jpeg.length < 4_900_000
    verify(jpeg.data, jpeg.sha256, jpeg.length)
    with pytest.raises(ValueError):
        verify(jpeg.data + b"x", jpeg.sha256, jpeg.length)


def test_quality_then_dimension_reduction():
    output = BytesIO()
    Image.effect_noise((1000, 1000), 100).save(output, format="PNG")
    jpeg = process(output.getvalue(), max_bytes=2000)
    assert jpeg.length < 2000
    with Image.open(BytesIO(jpeg.data)) as image:
        assert image.width < 1000
