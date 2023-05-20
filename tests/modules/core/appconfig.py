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
# import mod1
# mod1.call_count += 1

CALL_COUNT = 0

from pydantic import (
    BaseSettings,
    Field,
    PostgresDsn,
    RedisDsn,
)


class AppConfig(BaseSettings):
    environment: str = Field("localhost", env="APP_ENV,ENVIRONMENT")
    redis_url: RedisDsn = Field("redis://localhost:6379", env="REDIS_URL")
    postgres_url: PostgresDsn = Field(
        "postgresql://pguser:pgpass@localhost:5432/postgres?sslmode=disable",
        env="POSTGRES_URL",
    )

    class Config:
        # Important: environment variables will always take priority over over
        # values loaded from .env
        env_file = ".env"
        frozen = True
