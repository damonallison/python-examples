from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class EchoResult(BaseModel):
    echo: str
    updated_at: Optional[datetime]
