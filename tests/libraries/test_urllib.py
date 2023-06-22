import os
from urllib.parse import urlparse

def test_parse_url() -> None:
    u = "gs://host/to/file/file.txt"
    result = urlparse(u)

    assert result.scheme == "gs"
    assert result.netloc == "host"
    assert result.path == "/to/file/file.txt"

    assert os.path.dirname(u) == "gs://host/to/file"

def test_parse_file() -> None:
    u = "/path/to/file/file.txt"
    result = urlparse(u)

    assert result.scheme == ""
    assert result.netloc == ""
    assert result.path == "/path/to/file/file.txt"
