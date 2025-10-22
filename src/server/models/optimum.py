from typing import Dict, List, Union, Any
from pydantic import BaseModel, Field

# converted from https://huggingface.co/docs/transformers/main_classes/tokenizer
class PreTrainedTokenizerConfig(BaseModel):
    """
    Configuration for tokenizer.
    """

    text: Union[str, List[str], List[List[str]]] | None = Field(
        default=None,
        description=(
            "The sequence or batch of sequences to be encoded. Each sequence can be a string "
            "or a list of strings (pretokenized string). If the sequences are provided as "
            "list of strings (pretokenized), you must set is_split_into_words=True (to lift "
            "the ambiguity with a batch of sequences)."
        )
    )

    text_pair: Union[str, List[str], List[List[str]]] | None = Field(
        default=None,
        description=(
            "The sequence or batch of sequences to be encoded. Each sequence can be a string "
            "or a list of strings (pretokenized string). If the sequences are provided as "
            "list of strings (pretokenized), you must set is_split_into_words=True (to lift "
            "the ambiguity with a batch of sequences)."
        )
    )

    text_target: Union[str, List[str], List[List[str]]] | None = Field(
        default=None,
        description=(
            "The sequence or batch of sequences to be encoded as target texts. Each sequence can be "
            "a string or a list of strings (pretokenized string). If the sequences are provided as "
            "list of strings (pretokenized), you must set is_split_into_words=True (to lift the "
            "ambiguity with a batch of sequences)."
        )
    )

    text_pair_target: Union[str, List[str], List[List[str]]] | None = Field(
        default=None,
        description=(
            "The sequence or batch of sequences to be encoded as target texts. Each sequence can be "
            "a string or a list of strings (pretokenized string). If the sequences are provided as "
            "list of strings (pretokenized), you must set is_split_into_words=True (to lift the "
            "ambiguity with a batch of sequences)."
        )
    )

    add_special_tokens: bool = Field(
        default=True,
        description=(
            "Whether or not to add special tokens when encoding the sequences. This will use the "
            "underlying PretrainedTokenizerBase.build_inputs_with_special_tokens function, which "
            "defines which tokens are automatically added to the input ids. This is useful if you "
            "want to add bos or eos tokens automatically."
        )
    )

    padding: Union[bool, str] = Field(
        default=False,
        description=(
            "Activates and controls padding. Accepts the following values: True or 'longest': Pad to the "
            "longest sequence in the batch (or no padding if only a single sequence is provided). "
            "'max_length': Pad to a maximum length specified with the argument max_length or to the maximum "
            "acceptable input length for the model if that argument is not provided. False or 'do_not_pad' "
            "(default): No padding (i.e., can output a batch with sequences of different lengths)."
        )
    )

    truncation: Union[bool, str] = Field(
        default=False,
        description=(
            "Activates and controls truncation. Accepts the following values: True or 'longest_first': "
            "Truncate to a maximum length specified with the argument max_length or to the maximum acceptable "
            "input length for the model if that argument is not provided. This will truncate token by token, "
            "removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) "
            "is provided. 'only_first': Truncate to a maximum length specified with the argument max_length or to "
            "the maximum acceptable input length for the model if that argument is not provided. This will only "
            "truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided. "
            "'only_second': Truncate to a maximum length specified with the argument max_length or to the maximum "
            "acceptable input length for the model if that argument is not provided. False or 'do_not_truncate' "
            "(default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum "
            "admissible input size)."
        )
    )

    max_length: int | None = Field(
        default=None,
        description=(
            "Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to "
            "None, this will use the predefined model maximum length if a maximum length is required by one of the "
            "truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/"
            "padding to a maximum length will be deactivated."
        )
    )

    stride: int = Field(
        default=0,
        description=(
            "If set to a number along with max_length, the overflowing tokens returned when return_overflowing_tokens=True "
            "will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated "
            "and overflowing sequences. The value of this argument defines the number of overlapping tokens."
        )
    )

    is_split_into_words: bool = Field(
        default=False,
        description=(
            "Whether or not the input is already pre-tokenized (e.g., split into words). If set to True, the tokenizer "
            "assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. "
            "This is useful for NER or token classification."
        )
    )

    pad_to_multiple_of: int | None = Field(
        default=None,
        description=(
            "If set will pad the sequence to a multiple of the provided value. Requires padding to be activated. "
            "This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability "
            ">= 7.5 (Volta)."
        )
    )

    padding_side: str | None = Field(
        default=None,
        description=(
            "The side on which the model should have padding applied. Should be selected between ['right', 'left']. "
            "Default value is picked from the class attribute of the same name."
        )
    )

    return_tensors: str = Field(
        default="pt",
        description=(
            "If set, will return tensors instead of list of python integers. Acceptable values are: 'tf': Return TensorFlow "
            "tf.constant objects. 'pt': Return PyTorch torch.Tensor objects. 'np': Return Numpy np.ndarray objects."
        )
    )

    return_token_type_ids: bool | None  = Field(
        default=None,
        description=(
            "Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific "
            "tokenizer’s default, defined by the return_outputs attribute. What are token type IDs?"
        )
    )

    return_attention_mask: bool | None  = Field(
        default=None,
        description=(
            "Whether to return the attention mask. If left to the default, will return the attention mask according to the "
            "specific tokenizer’s default, defined by the return_outputs attribute. What are attention masks?"
        )
    )

    return_overflowing_tokens: bool = Field(
        default=False,
        description=(
            "Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) "
            "is provided with truncation_strategy = longest_first or True, an error is raised instead of returning overflowing tokens."
        )
    )

    return_special_tokens_mask: bool = Field(
        default=False,
        description=(
            "Whether or not to return special tokens mask information."
        )
    )

    return_offsets_mapping: bool = Field(
        default=False,
        description=(
            "Whether or not to return (char_start, char_end) for each token. This is only available on fast tokenizers inheriting from "
            "PreTrainedTokenizerFast, if using Python’s tokenizer, this method will raise NotImplementedError."
        )
    )

    return_length: bool = Field(
        default=False,
        description=(
            "Whether or not to return the lengths of the encoded inputs."
        )
    )

    verbose: bool = Field(
        default=True,
        description=(
            "Whether or not to print more information and warnings."
        )
    )

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