"""FastAPI Example

FastAPI is a wrapper on top of Starlette which provides:

* High performance. Based on Starlette / Uvicorn.
* Strong request / response typing w/ validation (via Pydantic).
* Automatic documentation generation
  * /docs - open api (swagger) documentation
  * /redoc - redoc documentation
  * /openapi.json - the open api generated schema

Pydantic
--------

FastAPI is based around Pydantic and strongly favors using type hints. The
benefits of pydantic and strong typing include:

* Requirements definition / documentation
  * Request path parmeters, headers, bodies, etc.
* Data conversion
  * Converts path parameters and request bodies into stronly typed objects.
* Validation
  * Provides data validation and generates specific error messages.

It goes without saying, but *always* use type hints and strongly type your code.

Concurrency
-----------

You can mix `async def` and `def` methods. The general guidance is:

* If you need to call `await` with a 3rd party libarry, use `async def`
* If 3rd party libraries do *not* support await, use `def`
* If you *don't* communicate with 3rd party libraries, use `async def`
* If you don't know what to do, use `def`

* Concurrency (async) is great for I/O bound operations (web).
* Parallelism (multiprocessor) is great for CPU bound operations.

FastAPI is based on AnyIO, which makes it compatible with `asyncio` (python's
standard libarry) and `trio`.
"""

from typing import List, Optional

from datetime import datetime
import logging
import uuid
import fastapi
import time

from fastapi import Body, Path, Query

from starlette_context import middleware, plugins

from .models import echo, estimate
from .models.context import RequestContext
from . import bundle_plugin


logger: logging.Logger

app = fastapi.FastAPI()

# Add context middleware
# app.add_middleware(middleware.RawContextMiddleware,
#     plugins=[
#     ],
# )

app.add_middleware(
    middleware.ContextMiddleware,
    plugins=[
        plugins.request_id.RequestIdPlugin(force_new_uuid=True),
        bundle_plugin.BundleIDPlugin(),
    ],
)


# Application events - startup / shutdown
@app.on_event("startup")
async def startup_logging():
    global logger
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("app.startup: started logging")


@app.on_event("shutdown")
async def shutdown():
    logger.info("app.shutdown: shutting down")


# Middleware
@app.middleware("http")
async def time_request(request: fastapi.applications.Request, call_next):
    """An example middleware."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = str(duration)
    logger.info(f"X-Process-Time {duration} {request.url}")
    return response


@app.post("/debug")
async def get_debug() -> dict:
    with RequestContext.new() as cc:
        RequestContext.add("inside", "true")
        print(RequestContext.all())
        with RequestContext.new() as cc2:
            RequestContext.add("deep inside", "true")
            print(RequestContext.all())
        print(RequestContext.all())
    print(RequestContext.all())
    return RequestContext.all()


# Note that path operations are evaluated in the order they are declared.
#
# This matters when paths could match two routes. For example, if you want to
# specify a /users/me special route and also a generic /users/{user_id} route,
# you'd need to specify the /me route before the /{user_id} route.
#
# * /users/me
# * /users/{user_id}
#
# Adding `response_model` adds the response object when
# generating documentation. If `response_model` is not specified, the
# documentation will show `null` when rendering the operation's response.
@app.get(
    "/",
    name="Root",
    description="Returns a simple 'hello world' data structure",
    response_model=echo.EchoResult,
)
async def get_root() -> echo.EchoResult:
    j = {"echo": "hello world", "updated_at": datetime.utcnow()}
    # time.sleep(0.1)
    return echo.EchoResult(**j)


# Path parameters are automatically parsed and passed to the function as
# arguments with the same name as defined in the path. Here `e` is pased to the
# function. Type annotations within the path parameter itself (i.e., (e:str})
# are optional (but nice to have).
#
# Other function parameters that are *not* part of the path are assumed to be
# query parameters. Here, `q` is an optional query parameter. If the function
# parameter is *not* `Optional`, FastAPI will validate it exists.
#
# The types `Path` and `Query` can be used to further validate path and query
# parameter values. The first parameter to Path and Query specify the default
# value. To specify the parameter is required with no default, use
# `...`.
#
# Here, `e` is required, q is optional with a default value of None.
#
# Query and Path also allow you to add metadata to the parameter which will be
# rendered with the documentation. Note that redoc uses both title and
# description, and /docs will only use description.
@app.get(
    "/echo/{e:str}",
    name="Echo",
    description="Echos back the incoming route parameter as well as an optional 'q' querystring parameter.",
    response_model=echo.EchoResult,
)
def get_echo(
    e: str = Path(..., description="The value to echo back", min_length=2),
    q: Optional[str] = Query(
        None,
        title="An optional querystring parameter",
        description="When present, echo will also echo this string back.",
    ),
) -> echo.EchoResult:
    if e == "test":
        return responses.Response(
            status_code=fastapi.status.HTTP_301_MOVED_PERMANENTLY,
            headers={"X-Test-Header": "test"},
        )
    if q is not None:
        e += f" {q}"

    return echo.EchoResult(echo=e, updated_at=datetime.utcnow())


@app.get(
    "/estimate/{id}",
    response_model=estimate.EstimateResponse,
)
def get_estimate(id: uuid.UUID) -> estimate.EstimateResponse:
    return estimate.EstimateResponse(estimates=[estimate.Estimate(id=id, value=100.0)])


# A function parameter that is a Pydantic BaseModel is assumed to be sent in the
# request body. It is possible to get access to the raw request body and not use
# FastAPI's defaults.
#
# See:
# https://fastapi.tiangolo.com/tutorial/body-multiple-params/#singular-values-in-body
#
# Example:
# curl -X POST -d '{"times": 10}' --silent http://localhost:8000/estimate
@app.post(
    "/estimate",
    response_model=estimate.EstimateResponse,
    status_code=fastapi.status.HTTP_201_CREATED,
)
def post_estimate(req: estimate.EstimateRequest) -> estimate.EstimateResponse:
    print(f"received {req}")
    ests: List[estimate.Estimate] = []
    for i in range(req.times):
        ests.append(estimate.Estimate(id=uuid.uuid4(), value=i * i))
    return estimate.EstimateResponse(estimates=ests)


# You can tell FastAPI to expect a parameter embedded within the top level of
# JSON.
#
# Here, both "estimate" and "backup_estimate" are expected to be child objects
# in the request body.
#
# {
#   "original" {"id": 1, "value": 1.0},
#   "backup" {"id": 1, "value": 2.0}
# }
@app.post(
    "/v2/estimate",
    response_model=estimate.EstimateResponse,
    status_code=fastapi.status.HTTP_201_CREATED,
)
def post_estimate(
    original: estimate.EstimateRequest = Body(None, embed=True),
    backup: estimate.EstimateRequest = Body(None, embed=True),
) -> estimate.EstimateResponse:
    ests: List[estimate.Estimate] = []
    if original is None and backup is None:
        raise ValueError("original or backup required")
    if original is None:
        original = backup
    for i in range(original.times):
        ests.append(estimate.Estimate(id=uuid.uuid4(), value=i * i))
    return estimate.EstimateResponse(estimates=ests)


# By declaring a `request` parameter, FastAPI will pass the incoming request.
# This gives you raw access to the incoming request.
@app.post("/debug", status_code=fastapi.status.HTTP_201_CREATED)
async def debug(request: fastapi.Request):
    j: dict = await request.json()
    print(f"request: {request.base_url} body: {j} client: {request.client.host}")
    if j.get("name", None) is None:
        raise fastapi.exceptions.HTTPException(
            fastapi.status.HTTP_400_BAD_REQUEST, "item not found"
        )

    # Sending back a custom response
    # response = fastapi.Response(
    #     json.dumps({"name": "damon"}),

    #     fastapi.status.HTTP_201_CREATED,
    #     media_type="application/json",
    # )
    # return response
    return {"name": j.get("name")}


# async def debug(request: fastapi.Request):
#     print(f"request: {request.base_url} {await request.json()}")

#     # Sending back a custom response
#     response = fastapi.Response(
#         json.dumps({"name": "damon"}),
#         fastapi.status.HTTP_201_CREATED,
#         media_type="application/json",
#     )
#     await response(request.scope, request.receive, request.send)
#     # return fastapi.responses.JSONResponse({"name": "damon"})
