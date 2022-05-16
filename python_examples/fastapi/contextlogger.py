from typing import Any, Dict
from contextvars import ContextVar
from uuid import uuid4

from fastapi.middleware import Middleware
from fastapi.requests import Request

_CONTEXT_DICT_KEY = "context_dict"
_context_dict: ContextVar[Dict[str, Any]] = ContextVar(_CONTEXT_DICT_KEY, default={})
