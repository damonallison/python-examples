"""Examples showing usage of the inspect module

inspect - inspect live objects
https://docs.python.org/3.8/library/inspect.html
"""

from typing import Any

from abc import ABC, abstractmethod
import inspect


class BasePredictor(ABC):
    """Base predictor class.

    Note the type of ABC is ABCMeta, therefore don't use multiple inheritance as
    it may lead to metaclass conflicts.
    """

    @abstractmethod
    def predict(self, arg: Any) -> Any:
        pass

    @abstractmethod
    def healthcheck(self, arg: Any) -> Any:
        pass

    def payload_type(self) -> type:
        sig = inspect.signature(self.predict)
        arg_param = sig.parameters.get("arg")
        assert arg_param.annotation
        return arg_param.annotation

    def return_type(self) -> type:
        sig = inspect.signature(self.predict)
        assert sig.return_annotation
        return sig.return_annotation


class DefaultPredictor(BasePredictor):
    def predict(self, arg: int) -> int:
        return arg + 10

    def healthcheck(self, arg: int) -> int:
        return arg + 100


def test_default_inspection() -> None:
    """Verifies inspect inspects the derived, not base, class."""
    d = DefaultPredictor()

    assert d.return_type() == int
    assert d.payload_type() == int

    assert d.predict(10) == 20
    assert d.healthcheck(10) == 110
