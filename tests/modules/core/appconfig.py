# A "global" variable.
#
# Note that python does not have truly *global* variables, only module level
# variables.
#
# Anyone wanting to access the variable must import a reference to the *module*.
# Importing the symbol (from mod1 import call_count) will create a *new*
# variable with the initial value of `call_count`. They will be different,
# independent variables!
#
# Importing a reference to the module will *not* create a copy of the call count
# variable
#
# import mod1
# mod1.CALL_COUNT += 1
#
# Importing CALL_COUNT *directly* from another module will create a copy of the
# variable in the importing module's namespace

import pydantic
import pydantic_settings

CALL_COUNT = 0

class AppConfig(pydantic_settings.BaseSettings):
    # model_config = pydantic_settings.SettingsConfigDict(
    #     # by default, pydantic will only validate on initial model creation.
    #     # This will re-validate each field as they are being set.
    #     validate_assignment=True,
    #     # Important: environment variables will always take priority over over
    #     # values loaded from .env. Directly setting values on models takes
    #     # priority over environment variables.
    #     #
    #     # Priority:
    #     #
    #     # * Values specifically set on models
    #     # * Environment variables
    #     # * env_file
    #     # env_file=".env",
    #     # faux immutability
    #     frozen=True,
    #     # case_sensitive is False by default. You can set to True if you want
    #     # case sensitivity.
    #     case_sensitive=False,
    # )
    environment: str = pydantic.Field(
        "localhost",
        alias=pydantic.AliasChoices("APP_ENV", "ENVIRONMENT"),
    )
    redis_url: pydantic.RedisDsn = pydantic.Field(
        "redis://localhost:6379",
    )
    postgres_url: pydantic.PostgresDsn = pydantic.Field(
        "postgresql://pguser:pgpass@localhost:5432/postgres?sslmode=disable",
    )

    # Complex types can be read from env vars as json strings
    # export ACTIVE_METROS="[1, 2, 3]"
    #
    # JSON is only parsed for top level fields.
    active_metros: list[str] = pydantic.Field(default_factory=list)
