from pydantic import BaseModel


class GenerateResponse(BaseModel):
    response: str
    model_used: str
    input_type: str
