# coding=utf-8
# Copyright 2020 Google and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tokenization class for model PEGASUS."""


import os
from shutil import copyfile
from typing import List, Optional, Tuple

from ...file_utils import add_start_docstrings, is_sentencepiece_available
from ...tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING, BatchEncoding
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging


if is_sentencepiece_available():
    from .tokenization_pegasus import PegasusTokenizer
else:
    PegasusTokenizer = None


logger = logging.get_logger(__name__)


SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {"google/pegasus-xsum": "https://cdn.huggingface.co/google/pegasus-xsum/spiece.model"},
    "tokenizer_file": {"google/pegasus-xsum": "https://cdn.huggingface.co/google/pegasus-xsum/tokenizer.json"},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "google/pegasus-xsum": 512,
}


class PegasusTokenizerFast(PreTrainedTokenizerFast):
    r"""
    Construct a "fast" PEGASUS tokenizer (backed by HuggingFace's `tokenizers` library). Based on `Unigram
    <https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end of
                sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask_2>"`):
            The token used for masking single token values. This is the token used when training this model with masked
            language modeling (MLM). This is the token that the PEGASUS encoder will try to predict during pretraining.
            It corresponds to `[MASK2]` in `PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive
            Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__.
        mask_token_sent (:obj:`str`, `optional`, defaults to :obj:`"<mask_1>"`):
            The token used for masking whole target sentences. This is the token used when training this model with gap
            sentences generation (GSG). This is the sentence that the PEGASUS decoder will try to predict during
            pretraining. It corresponds to `[MASK1]` in `PEGASUS: Pre-training with Extracted Gap-sentences for
            Abstractive Summarization <https://arxiv.org/pdf/1912.08777.pdf>`__.
        additional_special_tokens (:obj:`List[str]`, `optional`):
            Additional special tokens used by the tokenizer. If no additional_special_tokens are provided <mask_2> and
            <unk_2, ..., unk_102> are used as additional special tokens corresponding to the `original PEGASUS
            tokenizer
            <https://github.com/google-research/pegasus/blob/939830367bcf411193d2b5eca2f2f90f3f9260ca/pegasus/ops/pretrain_parsing_ops.cc#L66>`__
            that uses the tokens 2 - 104 only for pretraining
    """
    offset = 103  # entries 2-104 are only used for pretraining
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    slow_tokenizer_class = PegasusTokenizer
    model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        additional_special_tokens=None,
        **kwargs
    ):
        if additional_special_tokens is not None:
            assert isinstance(
                additional_special_tokens, list
            ), f"additional_special_tokens should be of type {type(list)}, but is {type(additional_special_tokens)}"

            additional_special_tokens_extended = (
                ([mask_token_sent] + additional_special_tokens)
                if mask_token_sent not in additional_special_tokens
                else additional_special_tokens
            )
            # fill additional tokens with ..., <unk_token_102> in case not all additional tokens are already taken
            additional_special_tokens_extended += [
                f"<unk_{i}>" for i in range(len(additional_special_tokens_extended), self.offset - 1)
            ]

            if len(set(additional_special_tokens_extended)) != len(additional_special_tokens_extended):
                raise ValueError(
                    f"Please make sure that the provided additional_special_tokens do not contain an incorrectly shifted list of <unk_x> tokens. Found {additional_special_tokens_extended}."
                )
            additional_special_tokens = additional_special_tokens_extended
        else:
            additional_special_tokens = [mask_token_sent]
            additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]

        super().__init__(
            vocab_file,
            tokenizer_file=tokenizer_file,
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.vocab_file = vocab_file

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special

        assert all_special_ids == set(
            range(len(self.additional_special_tokens) + 3)
        ), f"There should be 3 special tokens: mask_token, pad_token, and eos_token + {len(self.additional_special_tokens)} additional_special_tokens, but got {all_special_ids}"

        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """
        Build model inputs from a sequence by adding eos to the end. no bos token is added to the front.

        - single sequence: ``X </s>``
        - pair of sequences: ``A B </s>`` (not intended use)

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        tgt_texts: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        return_tensors: str = None,
        truncation=True,
        padding="longest",
        **unused,
    ) -> BatchEncoding:
        if "" in src_texts:
            raise ValueError(f"found empty string in src_texts: {src_texts}")
        tokenizer_kwargs = dict(
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )
        model_inputs: BatchEncoding = self(src_texts, **tokenizer_kwargs)
        if tgt_texts is None:
            return model_inputs
        if max_target_length is not None:
            tokenizer_kwargs["max_length"] = max_target_length
        labels: BatchEncoding = self(tgt_texts, **tokenizer_kwargs)["input_ids"]
        model_inputs["labels"] = labels
        return model_inputs

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
