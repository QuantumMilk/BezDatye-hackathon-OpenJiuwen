import os

DEFAULT_MODEL = "qwen3.8-27b"
MODELS = frozenset({
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
})


def budget_units(argument: str | None = None) -> int | None:
    value = argument if argument is not None else os.getenv("BUDGET")
    if value is None:
        return None
    try:
        tokens = int(value)
    except ValueError as error:
        raise ValueError("Budget must be a positive integer") from error
    if tokens <= 0:
        raise ValueError("Budget must be a positive integer")
    return 10 * tokens
