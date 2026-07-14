
from typing import List, Union
from pydantic import BaseModel, Field



class RerankerConfig(BaseModel):

    query: str = Field(
        default=None,
        description=(
            "Phrase to compare documents to."
        )
    )
    documents:  List[str] = Field(
        default=None,
        description=(
            "Documents to rank."
        )
    )

    prefix: str = Field(
        default='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
        description=(
            "Text to append to start of query. This is model specific."
        )
    )

    suffix: str = Field(
        default="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        description=(
            "Text to append to end of query. This is model specific and configured for Qwen3-Rerank tokenizer."
        )
    )

    instruction: str = Field(
        default="Given a search query, retrieve relevant passages that answer the query",
        description=(
            "Prompt command delivered to the model."
        )
    )

    max_length: int = Field(
        default=1024,
        description=(
            "Maximum sequence length for tokenization."
        )
    )