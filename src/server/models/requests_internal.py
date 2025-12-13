from typing import List, Optional

from pydantic import BaseModel


class OpenArcBenchRequest(BaseModel):
    model: str
    input_ids: List[int]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

