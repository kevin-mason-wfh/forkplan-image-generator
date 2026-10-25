from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

from PIL import Image, ImageOps


MAX_JPEG_BYTES = 4_900_000


@dataclass(frozen=True)
class Jpeg:
    data: bytes
    sha256: str
    length: int


def fingerprint(data: bytes) -> Jpeg:
    return Jpeg(data, sha256(data).hexdigest(), len(data))


def verify(data: bytes, digest: str, length: int) -> None:
    actual = fingerprint(data)
    if actual.sha256 != digest or actual.length != length or length >= MAX_JPEG_BYTES:
        raise ValueError("JPEG integrity check failed")


def process(original: bytes, max_bytes: int = MAX_JPEG_BYTES) -> Jpeg:
    if max_bytes < 1024 or max_bytes > MAX_JPEG_BYTES:
        raise ValueError("Invalid JPEG byte limit")
    with Image.open(BytesIO(original)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
    image = background
    while True:
        for quality in (95, 85, 75, 65, 55, 45, 35):
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            data = output.getvalue()
            if len(data) < max_bytes:
                return fingerprint(data)
        if min(image.size) <= 32:
            raise ValueError("Cannot encode JPEG below byte limit")
        image = image.resize(tuple(max(1, int(d * 0.8)) for d in image.size),
                             Image.Resampling.LANCZOS)
