from fastapi import FastAPI, status, responses
from pydantic import BaseModel

from typing import Optional, List
from datetime import datetime

class Estimate(BaseModel):
    value: float

class EstimateRequest(BaseModel):
    id: int
    times: int = 1

class EstimateResponse(BaseModel):
    estimates: List[Estimate]

class EchoResult(BaseModel):
    echo: str
    updated_at: Optional[datetime]

app = FastAPI()

@app.get("/",
    response_model=EchoResult,
    status_code=status.HTTP_200_OK,
)
def get_root() -> EchoResult:
    j = {
        "echo": "hello world",
        "updated_at": datetime.utcnow()
    }
    return EchoResult(**j)

@app.get("/echo/{echo}", response_model=EchoResult)
def get_echo(echo: str) -> EchoResult:
    if echo == "test":
        return responses.Response(status_code=status.HTTP_301_MOVED_PERMANENTLY, headers={"X-Test-Header": "test"})
    return EchoResult(echo=echo)

@app.get("/estimate/{id}", response_model=EstimateResponse)
def get_estimates(id: int, times: int = 1):
    ests: List[Estimate] = []
    for i in range(times):
        ests.append(Estimate(value=i))
    return EstimateResponse(estimates=ests)

@app.post("/estimate",
    response_model=EstimateResponse,
    status_code=status.HTTP_201_CREATED,
)
def post_estimate(req: EstimateRequest) -> EstimateResponse:
    ests: List[Estimate] = []
    for i in range(req.times):
        ests.append(Estimate(value=i))
    return EstimateResponse(estimates=ests)

@app.post("/run", response_model=EstimateResponse)
def post_run(req: EstimateRequest) -> EstimateResponse:
    ests: List[Estimate] = []
    for i in range(req.times):
        ests.append(Estimate(value=i))
    return EstimateResponse(estimates=ests)
    # Load model
    # Run
    # Send back results



