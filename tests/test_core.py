import pytest

from astro_simclr import greet


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Ada", "Hello, Ada!"),
        ("  Ada  ", "Hello, Ada!"),
        ("", "Hello, world!"),
        ("  ", "Hello, world!"),
    ],
)
def test_greet(name: str, expected: str) -> None:
    assert greet(name) == expected
