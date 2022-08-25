import json

from typing import Any, Optional

from starlette.requests import Request
from starlette_context.plugins.base import Plugin

class BundleIDPlugin(Plugin):

    key = "bundle-context"

    async def process_request(self, request: Request) -> Optional[Any]:
        try:
            j = await request.json()
            if isinstance(j, dict):
                return j.get("bundle-id", None)
        except json.JSONDecodeError:
            pass
        return None

