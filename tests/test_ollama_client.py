"""
Tests for app.services.ollama_client.generate().

Every test injects a mocked httpx.AsyncClient directly (dependency
injection) — no real network call, no patching of global state.
"""

import base64
from unittest.mock import patch

import httpx
import pytest

from app.services.ollama_client import (
    generate,
    OLLAMA_TEXT_MODEL,
    OLLAMA_VISION_MODEL,
    OLLAMA_BASE_URL,
)


class TestGenerateRouting:
    async def test_text_only_uses_text_model(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"response": "Paris"}
        )

        result = await generate(
            prompt="What is the capital of France?",
            image_bytes=None,
            client=mock_http_client,
        )

        assert result["model_used"] == OLLAMA_TEXT_MODEL
        assert result["input_type"] == "text"
        assert result["response"] == "Paris"

    async def test_image_and_text_uses_vision_model(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"response": "A dog sitting on grass"}
        )

        result = await generate(
            prompt="What's in this image?",
            image_bytes=b"fake-jpeg-bytes",
            client=mock_http_client,
        )

        assert result["model_used"] == OLLAMA_VISION_MODEL
        assert result["input_type"] == "image+text"
        assert result["response"] == "A dog sitting on grass"

    async def test_text_only_payload_has_no_images_field(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"response": "Paris"}
        )

        await generate(prompt="hello", image_bytes=None, client=mock_http_client)

        _, kwargs = mock_http_client.post.call_args
        assert "images" not in kwargs["json"]

    async def test_image_payload_base64_encodes_the_image(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"response": "A cat"}
        )
        raw_bytes = b"\xff\xd8\xff-fake-jpeg-header"

        await generate(
            prompt="what is this", image_bytes=raw_bytes, client=mock_http_client
        )

        _, kwargs = mock_http_client.post.call_args
        sent_images = kwargs["json"]["images"]
        assert sent_images == [base64.b64encode(raw_bytes).decode("utf-8")]

    async def test_stream_is_always_false(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"response": "Paris"}
        )

        await generate(prompt="hello", client=mock_http_client)

        _, kwargs = mock_http_client.post.call_args
        assert kwargs["json"]["stream"] is False

    async def test_posts_to_correct_ollama_endpoint(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"response": "Paris"}
        )

        await generate(prompt="hello", client=mock_http_client)

        args, _ = mock_http_client.post.call_args
        assert args[0] == f"{OLLAMA_BASE_URL}/api/generate"


class TestGenerateErrorHandling:
    async def test_propagates_http_status_error_on_ollama_failure(
        self, mock_http_client, make_ollama_response
    ):
        mock_http_client.post.return_value = make_ollama_response(
            {"error": "model not found"}, status_code=404
        )

        with pytest.raises(httpx.HTTPStatusError):
            await generate(prompt="hello", client=mock_http_client)

    async def test_propagates_request_error_when_ollama_unreachable(
        self, mock_http_client
    ):
        mock_http_client.post.side_effect = httpx.ConnectError(
            "Connection refused", request=httpx.Request("POST", "http://fake")
        )

        with pytest.raises(httpx.RequestError):
            await generate(prompt="hello", client=mock_http_client)


class TestGenerateDefaultClient:
    async def test_creates_and_closes_own_client_when_none_provided(
        self, make_ollama_response
    ):
        """Confirms the fallback path works when no client is injected —
        e.g. a standalone script calling generate() directly.
        """
        from unittest.mock import AsyncMock

        fake_response = make_ollama_response({"response": "Paris"})

        with patch("app.services.ollama_client.httpx.AsyncClient") as client_cls:
            entered_client = client_cls.return_value.__aenter__.return_value
            entered_client.post = AsyncMock(return_value=fake_response)
            client_cls.return_value.__aexit__.return_value = None

            result = await generate(prompt="hello", client=None)

            assert result["response"] == "Paris"
            client_cls.assert_called_once()
