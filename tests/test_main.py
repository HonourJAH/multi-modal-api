"""
Tests for the FastAPI routes in app.main.

app.main.generate is patched throughout, so these tests verify HTTP
status codes, response shapes, multipart handling, and error-to-status-code
mapping — the actual text-vs-vision routing logic is already covered in
test_ollama_client.py and isn't re-tested here.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def patched_generate():
    with patch("app.main.generate", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "response": "Paris",
            "model_used": "llama3.2",
            "input_type": "text",
        }
        yield mock_gen


@pytest.fixture
def client(patched_generate):
    with TestClient(app) as test_client:
        yield test_client


class TestGenerateEndpoint:
    def test_text_only_returns_200(self, client):
        response = client.post(
            "/generate", data={"prompt": "What is the capital of France?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["response"] == "Paris"
        assert body["model_used"] == "llama3.2"
        assert body["input_type"] == "text"

    def test_image_and_text_returns_200(self, client, patched_generate):
        patched_generate.return_value = {
            "response": "A dog",
            "model_used": "moondream",
            "input_type": "image+text",
        }
        files = {"image": ("dog.jpg", b"fake-bytes", "image/jpeg")}

        response = client.post(
            "/generate", data={"prompt": "What's in this image?"}, files=files
        )

        assert response.status_code == 200
        assert response.json()["input_type"] == "image+text"

    def test_image_only_uses_default_prompt(self, client, patched_generate):
        files = {"image": ("dog.jpg", b"fake-bytes", "image/jpeg")}

        response = client.post("/generate", files=files)

        assert response.status_code == 200
        _, kwargs = patched_generate.call_args
        assert kwargs["prompt"] == "Describe this image."

    def test_shared_http_client_is_passed_to_generate(self, client, patched_generate):
        client.post("/generate", data={"prompt": "hello"})

        _, kwargs = patched_generate.call_args
        assert kwargs["client"] is client.app.state.http_client

    def test_rejects_non_image_upload_with_400(self, client):
        files = {"image": ("notes.txt", b"just text", "text/plain")}

        response = client.post("/generate", data={"prompt": "hello"}, files=files)

        assert response.status_code == 400
        assert "must be an image" in response.json()["detail"]

    def test_rejects_upload_with_missing_content_type(self, client):
        files = {"image": ("mystery_file", b"some bytes", "")}

        response = client.post("/generate", data={"prompt": "hello"}, files=files)

        assert response.status_code == 400

    def test_ollama_http_error_returns_502(self, client, patched_generate):
        request = httpx.Request("POST", "http://fake-ollama/api/generate")
        fake_response = httpx.Response(status_code=404, request=request)
        patched_generate.side_effect = httpx.HTTPStatusError(
            "404 error", request=request, response=fake_response
        )

        response = client.post("/generate", data={"prompt": "hello"})

        assert response.status_code == 502
        assert "Ollama returned an error" in response.json()["detail"]

    def test_ollama_unreachable_returns_503(self, client, patched_generate):
        patched_generate.side_effect = httpx.ConnectError(
            "Connection refused", request=httpx.Request("POST", "http://fake")
        )

        response = client.post("/generate", data={"prompt": "hello"})

        assert response.status_code == 503
        assert "Could not reach Ollama" in response.json()["detail"]

    def test_missing_prompt_and_image_still_succeeds_via_default_prompt(
        self, client, patched_generate
    ):
        """No image, no prompt — prompt falls back to the default text,
        which gets sent to the text model. Not an error case; just an
        unusual but valid request.
        """
        response = client.post("/generate")

        assert response.status_code == 200
        _, kwargs = patched_generate.call_args
        assert kwargs["prompt"] == "Describe this image."
        assert kwargs["image_bytes"] is None


class TestHealthEndpoint:
    def test_health_check_returns_200_and_reports_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert "ollama_reachable" in body

    def test_health_check_reports_ollama_unreachable(self, client):
        # No real Ollama in the test environment, so the shared http_client
        # will genuinely fail to connect — confirms the check surfaces
        # real connectivity state rather than always returning True.
        response = client.get("/health")
        assert response.json()["ollama_reachable"] is False

    def test_health_check_reports_ollama_reachable_when_it_responds(self, client):
        from unittest.mock import AsyncMock

        fake_response = AsyncMock()
        fake_response.raise_for_status = lambda: None
        client.app.state.http_client.get = AsyncMock(return_value=fake_response)

        response = client.get("/health")
        assert response.json()["ollama_reachable"] is True


class TestAppStateLifecycle:
    def test_http_client_exists_on_app_state_after_startup(self):
        with TestClient(app) as test_client:
            assert hasattr(test_client.app.state, "http_client")
            assert test_client.app.state.http_client is not None
