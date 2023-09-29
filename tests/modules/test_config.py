# GOAL: Prove that we can patch a single config object and all references are updated
#
# How:
#   * Import Config from multiple modules
#   * Patch the actual config object
#   * Verify all instances read the patched value

from tests.modules.pkg1 import mod1
from tests.modules.pkg2 import mod2

from tests.modules.core import config
from tests.modules.core.appconfig import CALL_COUNT, AppConfig

from pytest import MonkeyPatch


def test_global_variable_import() -> None:
    """Because call_count is imported directly into this module,
    call_count actually points to a *new* variable in this module's
    namespace. It is *not* the same as the call_count variable in mod1"""

    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 0
    assert mod2.CALL_COUNT == 0

    mod1.CALL_COUNT = 1

    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 1
    assert mod2.CALL_COUNT == 0

    mod1.Mod1Calculator().add(2, 2)
    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 2
    assert mod2.CALL_COUNT == 0

    mod2.Mod2Calculator().add(2, 2)
    assert CALL_COUNT == 0
    assert mod1.CALL_COUNT == 2
    assert mod2.CALL_COUNT == 1


def test_config(monkeypatch: MonkeyPatch) -> None:
    assert mod1.Mod1Calculator().get_environment() == "localhost"
    assert mod2.Mod2Calculator().get_environment() == "localhost"

    with monkeypatch.context() as mp:
        # APP_ENV takes priority over ENVIRONMENT
        monkeypatch.setenv("APP_ENV", "development")
        monkeypatch.setenv("ENVIRONMENT", "staging")
        ac = AppConfig()
        assert ac.environment == "development"

        # monkeypatch the new config into the global environment
        mp.setattr("tests.modules.core.config", ac)

        # Because mod1 imported the config object into its module space, it will
        # *not* see the global change.
        assert mod1.Mod1Calculator().get_environment() == "localhost"

        # Because mod2 imported the core module, it *will* see the global
        # change.
        assert mod2.Mod2Calculator().get_environment() == "development"

    # Verify monkeypatch rolled back the global config change properly
    assert mod1.Mod1Calculator().get_environment() == "localhost"
    assert mod2.Mod2Calculator().get_environment() == "localhost"
