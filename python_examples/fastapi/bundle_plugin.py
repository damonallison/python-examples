import json

from typing import Any, Dict, Optional

from starlette.requests import Request
from starlette_context.plugins.base import Plugin


class BundleIDPlugin(Plugin):

    key = "bundle-context"

    async def process_request(self, request: Request) -> Optional[Any]:
        bundle_context: Dict[str, Any] = {}
        try:
            j = await request.json()
            if isinstance(j, dict):
                bundle_context["bundle-id"] = j.get("bundle-id", None)
        except json.JSONDecodeError:
            pass

        return bundle_context

