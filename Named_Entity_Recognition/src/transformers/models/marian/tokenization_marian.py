# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import json
import re
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece

from ...file_utils import add_start_docstrings
from ...tokenization_utils import BatchEncoding, PreTrainedTokenizer
from ...tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING


vocab_files_names = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "source_spm": {"Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/source.spm"},
    "target_spm": {"Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/target.spm"},
    "vocab": {"Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/vocab.json"},
    "tokenizer_config_file": {
        "Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/tokenizer_config.json"
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"Helsinki-NLP/opus-mt-en-de": 512}
PRETRAINED_INIT_CONFIGURATION = {}

# Example URL https://huggingface.co/Helsinki-NLP/opus-mt-en-de/resolve/main/vocab.json


class MarianTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Marian tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        source_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (:obj:`str`, `optional`):
            A string representing the source language.
        target_lang (:obj:`str`, `optional`):
            A string representing the target language.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (:obj:`int`, `optional`, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.

    Examples::

        >>> from transformers import MarianTokenizer
        >>> tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        >>> src_texts = [ "I am a small frog.", "Tom asked his teacher for advice."]
        >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
        >>> batch_enc = tok.prepare_seq2seq_batch(src_texts, tgt_texts=tgt_texts, return_tensors="pt")
        >>> # keys  [input_ids, attention_mask, labels].
        >>> # model(**batch) should work
    """

    vocab_files_names = vocab_files_names
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]
    language_code_re = re.compile(">>.+<<")  # type: re.Pattern

    def __init__(
        self,
        vocab,
        source_spm,
        target_spm,
        source_lang=None,
        target_lang=None,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
        **kwargs
    ):
        super().__init__(
            # bos_token=bos_token,  unused. Start decoding with config.decoder_start_token_id
            source_lang=source_lang,
            target_lang=target_lang,
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            **kwargs,
        )
        assert Path(source_spm).exists(), f"cannot find spm source {source_spm}"
        self.encoder = load_json(vocab)
        if self.unk_token not in self.encoder:
            raise KeyError("<unk> token must be in vocab")
        assert self.pad_token in self.encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.supported_language_codes: list = [k for k in self.encoder if k.startswith(">>") and k.endswith("<<")]
        self.spm_files = [source_spm, target_spm]

        # load SentencePiece model for pre-processing
        self.spm_source = load_spm(source_spm)
        self.spm_target = load_spm(target_spm)
        self.current_spm = self.spm_source

        # Multilingual target side: default to using first supported language code.

        self._setup_normalizer()

    def _setup_normalizer(self):
        try:
            from sacremoses import MosesPunctNormalizer

            self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
        except (ImportError, FileNotFoundError):
            warnings.warn("Recommended: pip install sacremoses.")
            self.punc_normalizer = lambda x: x

    def normalize(self, x: str) -> str:
        """Cover moses empty string edge case. They return empty list for '' input!"""
        return self.punc_normalizer(x) if x else ""

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def remove_language_code(self, text: str):
        """Remove language codes like <<fr>> before sentencepiece"""
        match = self.language_code_re.match(text)
        code: list = [match.group(0)] if match else []
        return code, self.language_code_re.sub("", text)

    def _tokenize(self, text: str) -> List[str]:
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.EncodeAsPieces(text)
        return code + pieces

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the encoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses target language sentencepiece model"""
        return self.spm_target.DecodePieces(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
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
        return_tensors: Optional[str] = None,
        truncation=True,
        padding="longest",
        **unused,
    ) -> BatchEncoding:
        if "" in src_texts:
            raise ValueError(f"found empty string in src_texts: {src_texts}")
        self.current_spm = self.spm_source
        src_texts = [self.normalize(t) for t in src_texts]  # this does not appear to do much
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

        self.current_spm = self.spm_target
        model_inputs["labels"] = self(tgt_texts, **tokenizer_kwargs)["input_ids"]
        self.current_spm = self.spm_source
        return model_inputs

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        save_dir = Path(save_directory)
        assert save_dir.is_dir(), f"{save_directory} should be a directory"
        save_json(
            self.encoder,
            save_dir / ((filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab"]),
        )

        for orig, f in zip(["source.spm", "target.spm"], self.spm_files):
            dest_path = save_dir / ((filename_prefix + "-" if filename_prefix else "") + Path(f).name)
            if not dest_path.exists():
                copyfile(f, save_dir / orig)

        return tuple(
            save_dir / ((filename_prefix + "-" if filename_prefix else "") + f) for f in self.vocab_files_names
        )

    def get_vocab(self) -> Dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state.update({k: None for k in ["spm_source", "spm_target", "current_spm", "punc_normalizer"]})
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d
        self.spm_source, self.spm_target = (load_spm(f) for f in self.spm_files)
        self.current_spm = self.spm_source
        self._setup_normalizer()

    def num_special_tokens_to_add(self, **unused):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
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


def load_spm(path: str) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(path)
    return spm


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Union[Dict, List]:
    with open(path, "r") as f:
        return json.load(f)
