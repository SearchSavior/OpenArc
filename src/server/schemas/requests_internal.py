from typing import List, Optional

from pydantic import BaseModel, model_validator


class OpenArcBenchRequest(BaseModel):
    model: str
    input_ids: Optional[List[int]] = None
    prompt: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

    @model_validator(mode="after")
    def exactly_one_input(self) -> "OpenArcBenchRequest":
        ids_ok = self.input_ids is not None and len(self.input_ids) > 0
        prompt_ok = self.prompt is not None and self.prompt != ""
        if ids_ok == prompt_ok:
            raise ValueError(
                "Provide exactly one of: input_ids (non-empty list) or prompt (non-empty string)."
            )
        return self
