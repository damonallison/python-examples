from typing import Dict, List, TypedDict

ParseQSResult = Dict[str, List[str]]
ConnectionDict = TypedDict(
    "ConnectionDict", {"name": str, "pass": str, "query": ParseQSResult}
)


class TestTyping:
    def test_typedef(self) -> None:
        pass
