from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request, status

from app.schemas import GenerateResponse
from app.services.ollama_client import generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One shared httpx.AsyncClient for the app's whole lifetime — reused
    # across every /generate call instead of opening a fresh connection
    # per request. Closed cleanly on shutdown.
    app.state.http_client = httpx.AsyncClient()
    yield
    await app.state.http_client.aclose()


app = FastAPI(
    title="Multi-Modal API",
    description="Accepts text and/or image input and routes to the appropriate Ollama model",
    lifespan=lifespan,
)


@app.post("/generate", response_model=GenerateResponse)
async def generate_route(
    request: Request,
    prompt: str = Form("Describe this image.", min_length=1),
    image: UploadFile | None = File(None),
):
    image_bytes = None

    if image is not None:
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file must be an image, got content-type: {image.content_type}",
            )
        image_bytes = await image.read()

    try:
        result = await generate(
            prompt=prompt,
            image_bytes=image_bytes,
            client=request.app.state.http_client,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ollama returned an error: {exc.response.status_code}",
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not reach Ollama. Is it running?",
        )

    return GenerateResponse(**result)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
