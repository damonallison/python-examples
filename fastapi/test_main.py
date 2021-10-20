from main import app
from fastapi.testclient import TestClient

import yappi


class TestMain:
    def test_app(self):
        yappi.set_clock_type("wall")
        yappi.start()

        # Run TestClient in a with block to run startup / shutdown handlers
        with TestClient(app=app) as client:
            for _ in range(20):
                resp = client.get("/")
                assert resp.status_code == 200

        yappi.stop()
        with open("./out.txt", "w") as f:
            # Filter to *just* local names. You cold also filter based on module
            # name). Anything in the `YFuncStat` object.
            yappi.get_func_stats(
                filter_callback=lambda x: "python-examples/fastapi" in x.full_name
            ).print_all(
                out=f,
                columns={
                    0: ("name", 200),  # The default is 36, which is way too short.
                    1: ("ncall", 5),
                    2: ("tsub", 8),
                    3: ("ttot", 8),
                    4: ("tavg", 8),
                },
            )
