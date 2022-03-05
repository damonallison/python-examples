# Allows you to reference a type as an annotation within the type declaration.
# For example, we define the attr `children` as a list of `Person`. In order to
# refer to ``Person`, we must import annotations.`
from __future__ import annotations
from typing import List
import pydantic


class Person(pydantic.BaseModel):
    name: str
    children: List[Person]
