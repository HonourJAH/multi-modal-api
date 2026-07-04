"""
Shared fixtures for the multi-modal API test suite.

No real Ollama server is ever contacted here. httpx.AsyncClient is injected
into ollama_client.generate() (dependency injection), so tests build a
lightweight fake response object and a mocked client directly, the same
approach used for MlflowClient in the MLOps Pipeline API's test suite.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


def make_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Fabricate an object shaped like httpx.Response.

    raise_for_status() actually raises httpx.HTTPStatusError for 4xx/5xx,
    matching real httpx behavior, so ollama_client's except blocks are
    exercised the same way they would be against a real server.
    """
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data

    if status_code >= 400:
        request = httpx.Request("POST", "http://fake-ollama/api/generate")
        error = httpx.HTTPStatusError(
            f"{status_code} error", request=request, response=response
        )
        response.raise_for_status.side_effect = error
    else:
        response.raise_for_status.return_value = None

    return response


@pytest.fixture
def make_ollama_response():
    """Factory fixture so individual tests can build custom responses."""
    return make_response


@pytest.fixture
def mock_http_client():
    """An AsyncMock standing in for httpx.AsyncClient.

    Tests configure client.post.return_value (or side_effect) directly.
    """
    client = AsyncMock(spec=httpx.AsyncClient)
    return client
