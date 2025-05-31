from pydantic import BaseModel
from typing import Optional

class PromptRequest(BaseModel):
    prompt: str
