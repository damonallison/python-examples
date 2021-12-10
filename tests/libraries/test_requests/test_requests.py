"""Tests showing the use of requests and requests-mock."""
import json
import pytest
import requests
from requests_mock import Mocker, ANY


class TestRequests:
    def test_wire_failures(self) -> None:
        """All exceptions that Requests raises derive from RequestException.

        If you want to get more granular, you could also catch one or more derived exceptions:

        * ConnectionError
        * HTTPError
        * Timeout
        * TooManyRedirects

        https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
        """

        with pytest.raises(requests.exceptions.RequestException):
            requests.get("http://23423iu4iijsdfajdfh.coms")

    def test_requests_mock_using_kwargs(self, requests_mock: Mocker) -> None:
        """Shows mocking out a GET call to any URL and returning a mocked response.

        https://requests-mock.readthedocs.io/en/latest/response.html#registering-responses

        Settable args:

        status_code: The HTTP status code to return. Defaults to 200.
        reason: The text reason text that accompanies the status_code.
        headers: A dictionary of headers to be included in the response.

        To specify the body:

        json: a python object that will be converted to a JSON string
        text: a unicode string
        context: a byte string
        body: a file-like object that contains a `.read()` function
        exc: an exception that will be raised rather than returning a response
        """

        # Mock a response using kwargs
        requests_mock.get(
            ANY,
            status_code=201,
            json={"a": "test"},
        )

        # Make the request
        resp = requests.get("http://damonallison.com", json={"hello": "world"})

        # Verify mocked response
        assert resp.status_code == 201
        assert resp.json() == {"a": "test"}
        assert resp.text == json.dumps(resp.json())
        assert resp.content == bytes(resp.text, encoding="utf-8")

    def test_requests_mock_list_of_responses(self, requests_mock: Mocker) -> None:
        """You can mock responses with a list of dictionaries.

        Each dictionary is parsed into a response object.

        If there are more requests sent than responses mocked, the last response
        will be used.
        """
        requests_mock.get(
            ANY,
            response_list=[
                {
                    "status_code": 201,
                    "text": "hello, first",
                },
                {
                    "status_code": 200,
                    "json": "hello, second",
                },
            ],
        )

        resp1 = requests.get("http://google.com", data="hi there")
        resp2 = requests.get("http://google.com", data="hi there")
        resp3 = requests.get("http://google.com", data="hi there again")

        assert resp1.status_code == 201
        assert resp1.text == "hello, first"

        assert resp2.status_code == 200
        assert resp2.json() == "hello, second"

        # NOTE: The last response was used
        assert resp3.status_code == 200
        assert resp3.json() == "hello, second"

    def test_requests_mock_verify_sent_requests(self, requests_mock: Mocker) -> None:
        """Shows using the mock to verify the requests sent."""
        requests_mock.get(ANY, text='"hello, world"', status_code=201)

        resp1 = requests.get(
            "http://shipt.com", data="some data", headers={"X-TEST": "damon"}
        )
        resp2 = requests.get("http://apple.com", data="more data")

        # Verify responses
        assert resp1.json() == "hello, world"
        assert resp1.text == '"hello, world"'
        assert resp1.content == b'"hello, world"'

        assert resp2.json() == "hello, world"

        # Verify requests sent
        assert requests_mock.call_count == 2

        req1 = requests_mock.request_history[0]
        assert req1.url == "http://shipt.com/"
        assert req1.hostname == "shipt.com"
        assert req1.text == "some data"
        assert "X-TEST" in req1.headers.keys()
        assert "damon" == req1.headers["X-TEST"]

        req2 = requests_mock.request_history[1]
