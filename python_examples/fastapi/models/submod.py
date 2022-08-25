from typing import Any

from starlette_context import context


def bundle_context() -> dict:
    context.data["test"] = "value"
    print(context.data)
    return context.data["bundle-context"]

def add(key: str, val: Any) -> None:
    bundle_context()[key] = val

def get(key: str) -> Any:
    return bundle_context().get(key, None)
