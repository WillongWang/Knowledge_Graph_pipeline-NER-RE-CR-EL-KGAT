# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

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

from typing import TYPE_CHECKING

from ...file_utils import _BaseLazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_auto": ["ALL_PRETRAINED_CONFIG_ARCHIVE_MAP", "CONFIG_MAPPING", "MODEL_NAMES_MAPPING", "AutoConfig"],
    "tokenization_auto": ["TOKENIZER_MAPPING", "AutoTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_auto"] = [
        "MODEL_FOR_CAUSAL_LM_MAPPING",
        "MODEL_FOR_MASKED_LM_MAPPING",
        "MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "MODEL_FOR_PRETRAINING_MAPPING",
        "MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING",
        "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "MODEL_MAPPING",
        "MODEL_WITH_LM_HEAD_MAPPING",
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForMaskedLM",
        "AutoModelForMultipleChoice",
        "AutoModelForNextSentencePrediction",
        "AutoModelForPreTraining",
        "AutoModelForQuestionAnswering",
        "AutoModelForSeq2SeqLM",
        "AutoModelForSequenceClassification",
        "AutoModelForTableQuestionAnswering",
        "AutoModelForTokenClassification",
        "AutoModelWithLMHead",
    ]

if is_tf_available():
    _import_structure["modeling_tf_auto"] = [
        "TF_MODEL_FOR_CAUSAL_LM_MAPPING",
        "TF_MODEL_FOR_MASKED_LM_MAPPING",
        "TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING",
        "TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING",
        "TF_MODEL_FOR_PRETRAINING_MAPPING",
        "TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING",
        "TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING",
        "TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING",
        "TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING",
        "TF_MODEL_MAPPING",
        "TF_MODEL_WITH_LM_HEAD_MAPPING",
        "TFAutoModel",
        "TFAutoModelForCausalLM",
        "TFAutoModelForMaskedLM",
        "TFAutoModelForMultipleChoice",
        "TFAutoModelForPreTraining",
        "TFAutoModelForQuestionAnswering",
        "TFAutoModelForSeq2SeqLM",
        "TFAutoModelForSequenceClassification",
        "TFAutoModelForTokenClassification",
        "TFAutoModelWithLMHead",
    ]

if is_flax_available():
    _import_structure["modeling_flax_auto"] = ["FLAX_MODEL_MAPPING", "FlaxAutoModel"]


if TYPE_CHECKING:
    from .configuration_auto import ALL_PRETRAINED_CONFIG_ARCHIVE_MAP, CONFIG_MAPPING, MODEL_NAMES_MAPPING, AutoConfig
    from .tokenization_auto import TOKENIZER_MAPPING, AutoTokenizer

    if is_torch_available():
        from .modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING,
            MODEL_FOR_MASKED_LM_MAPPING,
            MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            MODEL_FOR_PRETRAINING_MAPPING,
            MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
            MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            MODEL_MAPPING,
            MODEL_WITH_LM_HEAD_MAPPING,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForMultipleChoice,
            AutoModelForNextSentencePrediction,
            AutoModelForPreTraining,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForTableQuestionAnswering,
            AutoModelForTokenClassification,
            AutoModelWithLMHead,
        )

    if is_tf_available():
        from .modeling_tf_auto import (
            TF_MODEL_FOR_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_MASKED_LM_MAPPING,
            TF_MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
            TF_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
            TF_MODEL_FOR_PRETRAINING_MAPPING,
            TF_MODEL_FOR_QUESTION_ANSWERING_MAPPING,
            TF_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
            TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
            TF_MODEL_MAPPING,
            TF_MODEL_WITH_LM_HEAD_MAPPING,
            TFAutoModel,
            TFAutoModelForCausalLM,
            TFAutoModelForMaskedLM,
            TFAutoModelForMultipleChoice,
            TFAutoModelForPreTraining,
            TFAutoModelForQuestionAnswering,
            TFAutoModelForSeq2SeqLM,
            TFAutoModelForSequenceClassification,
            TFAutoModelForTokenClassification,
            TFAutoModelWithLMHead,
        )

    if is_flax_available():
        from .modeling_flax_auto import FLAX_MODEL_MAPPING, FlaxAutoModel

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
