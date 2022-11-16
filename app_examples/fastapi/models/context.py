from copy import deepcopy
from contextlib import contextmanager
from typing import Any, Optional

from starlette_context import context


class _Context():
    """A lightweight wrapper around starlette_context's context object.

    This class allows you to access and manipulate state associated with the
    current HTTP request.

    This can be used for things like request level logging. For example, adding
    variables to all logging statements (context based logging).

    It can, but should *not* be used for storing and accessing request
    level state from within code. Do *not* rely on this "global" request
    level state as an easy alternative to passing state between functions.

    Example:

            from context import RequestContext as ctx

            cc.add("var1", "test")
            with ctx.new() as cc:
                cc.add("var2", "value") # adds a new context variable within this scope only
                print(cc.all()) # prints var1 and var2
            print(cc.all()) # var 1 only

    """

    @contextmanager
    def new(self) -> dict[str, Any]:
        prev = deepcopy(context.data)
        try:
            yield context.data
        finally:
            context.data.clear()
            context.data.update(prev)

    def add(self, key: str, val: Any) -> None:
        context[key] = val

    def delete(self, key: str) -> None:
        del context[key]

    def get(self, key: str, default: Optional[Any]) -> Any:
        return context.get(key, default)

    def all(self) -> dict[str, Any]:
        return deepcopy(context)

RequestContext = _Context()
