from datetime import datetime
from pydantic import BaseModel
import pytest
import random
import redis
import string
from typing import Optional


class Person(BaseModel):
    fname: str
    lname: str
    iq: Optional[int] = None
    created_at: datetime


class TestRedis:
    def gen_rand(self, len: int = 5) -> str:
        return "".join(
            random.SystemRandom().choice(string.ascii_uppercase + string.digits)
            for _ in range(len)
        )

    @pytest.mark.skip(reason="Requires a running redis instance")
    def test_set_get(self) -> None:
        key = f"test_set_get.{self.gen_rand()}"
        p = Person(fname="damon", lname="allison", created_at=datetime.utcnow())

        with redis.Redis(host="localhost", port=6379, db=0) as r:
            r.set(key, p.json())
            b = r.get(key)
            assert b is not None
            p2 = Person.parse_raw(b)
            assert p == p2
