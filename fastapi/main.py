"""FastAPI Example

FastAPI is a wrapper on top of Starlette which provides:

* High performance. Based on Starlette / Uvicorn.
* Strong request / response typing w/ validation (via Pydantic).
* Automatic documentation generation
  * /docs - open api (swagger) documentation
  * /redoc - redoc documentation
  * /openapi.json - the open api generated schema

  by default).
*


"""
import uuid
import json
import fastapi
from fastapi import responses, Body

from typing import List, Optional
from datetime import datetime

from models import estimate
from models import echo


app = fastapi.FastAPI()

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
@app.get("/", response_model=echo.EchoResult)
async def get_root() -> echo.EchoResult:
    j = {"echo": "hello world", "updated_at": datetime.utcnow()}
    return echo.EchoResult(**j)


# Path parameters are automatically parsed and passed to the function as
# arguments with the same name as defined in the path. Here `e` is pased to the
# function. Type annotations within the path parameter itself (i.e., (e:str})
# are optional (but nice to have).
#
# Other function parameters that are *not* part of the path are assumed to be
# query parameters. Here, `q` is an optional query parameter. If the function
# parameter is *not* `Optional`, FastAPI will validate it exists.
@app.get(
    "/echo/{e:str}",
    response_model=echo.EchoResult,
)
def get_echo(e: str, q: Optional[str]) -> echo.EchoResult:
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
def get_estimates(id: uuid.UUID) -> estimate.EstimateResponse:
    ests: List[estimate.EstimateResponse] = []
    return estimate.EstimateResponse(estimates=[estimate.Estimate(id=id, value=100.0)])


# A function parameter that is a Pydantic BaseModel is assumed to be sent in the
# request body. It is possible to get access to the raw request body and not use
# FastAPI's defaults.
#
# See:
# https://fastapi.tiangolo.com/tutorial/body-multiple-params/#singular-values-in-body


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
