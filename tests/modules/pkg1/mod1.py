from tests.modules.core import config
from tests.modules.core.appconfig import CALL_COUNT


class Mod1Calculator:
    def add(self, x: int, y: int) -> int:
        global CALL_COUNT
        CALL_COUNT += 1
        return x + y

    def get_environment(self) -> str:
        return config.environment

    def get_call_count(self) -> int:
        return CALL_COUNT
