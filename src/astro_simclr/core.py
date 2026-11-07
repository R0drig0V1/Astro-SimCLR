"""Example module: replace it with the project's code."""


def greet(name: str) -> str:
    """Return a greeting for `name`; an empty or blank name greets the world."""
    name = name.strip()
    return f"Hello, {name or 'world'}!"
