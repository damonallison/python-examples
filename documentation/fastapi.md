# Fast API

* Built on
  * [starlette](https://github.com/encode/starlette)
  

## Documentation

Fast API automatically generates OpenAPI documentation
http://localhost:8000/docs
http://localhost:8000/redoc


```python
from starlette.requests import Request
from fastapi import APIRouter
from loguru import logger
from app.data_models.heartbeat import HearbeatResult

import time
import threading

router = APIRouter()


class TestRefresher:
    def __init__(self) -> None:
        self.iteration: int = 0
        self._start_refresh()

    def _start_refresh(self) -> None:
        logger.info(f"TestRefresher...Starting")

        def run_daemon_thread() -> None:
            # While loop on 1 minute sleep timer to refresh singleton class features
            while True:
                logger.info(f"TestRefresher...Refreshing ({self.iteration})")
                if self.iteration == 10:
                    raise ValueError("boom")
                self.iteration = self.iteration + 1
                time.sleep(1)

        # Initialize daemon thread for feature refresher
        t1 = threading.Thread(target=run_daemon_thread, daemon=True)
        t1.start()


@router.get("/features", name="feature refresh tester")
def get_features(request: Request) -> dict:
    try:
        request.app.state.tr
    except AttributeError as e:
        request.app.state.tr = TestRefresher()

    logger.info(f"request count == {len(request.state._state)}")
    logger.info(f"app count == {len(request.app.state._state)}")
    for k, v in request.state._state.items():
        logger.info(f"request key=={k} val=={v}")
    for k, v in request.app.state._state.items():
        logger.info(f"app key=={k} val=={v}")
    return {"hello", "world"}


@router.get("/health", response_model=HearbeatResult, name="heartbeat")
def get_hearbeat() -> HearbeatResult:
    """Return heartbeat dataclass"""
    heartbeat = HearbeatResult(is_alive=True)
    return heartbeat
```
