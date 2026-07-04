import base64
import os

import httpx

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.2")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")


REQUEST_TIMEOUT_SECONDS = 120.0


async def generate(
    prompt: str,
    image_bytes: bytes | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Route a prompt to the appropriate Ollama model based on input type.

    - Text only (image_bytes is None)      -> OLLAMA_TEXT_MODEL
    - Text + image (image_bytes provided)  -> OLLAMA_VISION_MODEL
    """
    if image_bytes is not None:
        model = OLLAMA_VISION_MODEL
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [base64.b64encode(image_bytes).decode("utf-8")],
            "stream": False,
        }
    else:
        model = OLLAMA_TEXT_MODEL
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

    if client is not None:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    else:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as default_client:
            response = await default_client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

    return {
        "response": data["response"],
        "model_used": model,
        "input_type": "image+text" if image_bytes is not None else "text",
    }
