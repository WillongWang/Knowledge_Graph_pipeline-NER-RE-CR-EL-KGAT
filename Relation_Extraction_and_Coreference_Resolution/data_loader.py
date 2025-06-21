import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

from utils import get_re_labels

logger = logging.getLogger(__name__)


#############################
# add entity markers
#############################


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        re_labels: (Optional) list. The re labels of the example.
    """

    def __init__(self, guid, words,
                 re_label_id=None,
                 head_entity_pos=None,
                 tail_entity_pos=None,
                 ):
        self.guid = guid
        self.words = words
        self.re_label_id = re_label_id

        self.head_entity_pos = head_entity_pos  # tuple, (s, e)
        self.tail_entity_pos = tail_entity_pos  # tuple, (s, e)

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids,
                 attention_mask,
                 token_type_ids,
                 re_label_id,
                 rel_position_ids=None,
                 head_entity_pos=None,
                 tail_entity_pos=None,
                 ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.re_label_id = re_label_id

        self.rel_position_ids = rel_position_ids

        self.head_entity_pos = head_entity_pos
        self.tail_entity_pos = tail_entity_pos

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ReProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.re_label2id = get_re_labels(args)

        self.input_text_file = 'semeval.txt'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, line in enumerate(lines):
            guid = "%s-%s" % (set_type, i)

            line = line.strip()
            if not line:
                continue

            line = json.loads(line)

            # 关系标签
            re_label = line.get("relation", "Other")
            re_label_id = self.re_label2id[re_label]

            # input_text
            words = line["token"]

            # 头实体的位置
            head_entity_pos = line["h"]["pos"]
            head_entity_mention = line["h"]["name"]
            assert " ".join(words[head_entity_pos[0]: head_entity_pos[1]]) == head_entity_mention

            # 尾实体的位置
            tail_entity_pos = line["t"]["pos"]
            tail_entity_mention = line["t"]["name"]
            if " ".join(words[tail_entity_pos[0]: tail_entity_pos[1]]) != tail_entity_mention:
                print(" ".join(words[tail_entity_pos[0]: tail_entity_pos[1]]))
                print(tail_entity_mention)

            assert " ".join(words[tail_entity_pos[0]: tail_entity_pos[1]]) == tail_entity_mention

            examples.append(
                InputExample(
                    guid,
                    words,
                    re_label_id=re_label_id,
                    head_entity_pos=head_entity_pos,
                    tail_entity_pos=tail_entity_pos,
                )
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(lines=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     set_type=mode)


processors = {
    "semeval10": ReProcessor,
}


def convert_examples_to_features(examples,
                                 max_seq_len,
                                 tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True,
                                 head_entity_token="[unused4]",
                                 tail_entity_token="[unused5]",
                                 head_entity_start_token="[unused0]",
                                 head_entity_end_token="[unused2]",
                                 tail_entity_start_token="[unused1]",
                                 tail_entity_end_token="[unused3]",
                                 span_identification_method="v1"
                                 ):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # 关系标签
        re_label_id = int(example.re_label_id)

        head_entity_start_pos = None
        tail_entity_start_pos = None
        head_entity_end_pos = None
        tail_entity_end_pos = None

        # Tokenize word by word (for RE task)
        tokens = []

        if span_identification_method == "v1":

            for i, word in enumerate(example.words):
                word_tokens = tokenizer.tokenize(word)

                if example.head_entity_pos[0] == i:
                    head_entity_start_pos = len(tokens)
                if example.tail_entity_pos[0] == i:
                    tail_entity_start_pos = len(tokens)

                tokens.extend(word_tokens)

                if example.head_entity_pos[1] == i + 1:
                    head_entity_end_pos = len(tokens)

                if example.tail_entity_pos[1] == i + 1:
                    tail_entity_end_pos = len(tokens)

        elif span_identification_method == "v2":
            tokens = []
            for i, word in enumerate(example.words):
                word_tokens = tokenizer.tokenize(word)

                if example.head_entity_pos[0] == i:
                    head_entity_start_pos = len(tokens)
                    tokens.append(head_entity_start_token)
                if example.tail_entity_pos[0] == i:
                    tail_entity_start_pos = len(tokens)
                    tokens.append(tail_entity_start_token)

                tokens.extend(word_tokens)

                if example.head_entity_pos[1] == i + 1:
                    tokens.append(head_entity_end_token)
                    head_entity_end_pos = len(tokens)

                if example.tail_entity_pos[1] == i + 1:
                    tokens.append(tail_entity_end_token)
                    tail_entity_end_pos = len(tokens)


        else:

            tokens = []
            for i, word in enumerate(example.words):
                word_tokens = tokenizer.tokenize(word)

                if example.head_entity_pos[0] == i:
                    head_entity_start_pos = len(tokens)
                    tokens.append(head_entity_token)
                    head_entity_end_pos = len(tokens)

                if example.head_entity_pos[0] < i < example.head_entity_pos[1]:
                    continue

                if example.tail_entity_pos[0] == i:
                    tail_entity_start_pos = len(tokens)
                    tokens.append(tail_entity_token)
                    tail_entity_end_pos = len(tokens)

                if example.tail_entity_pos[0] < i < example.tail_entity_pos[1]:
                    continue

                tokens.extend(word_tokens)

        # 如果句长太长，导致头尾实体不全，则删除
        if not head_entity_start_pos or not tail_entity_start_pos \
                or not head_entity_end_pos or not tail_entity_end_pos:
            continue

        rel_position_ids = [0] * len(tokens)
        rel_position_ids[head_entity_start_pos: head_entity_end_pos] = \
            [1, ] * (head_entity_end_pos - head_entity_start_pos)
        rel_position_ids[tail_entity_start_pos: tail_entity_end_pos] = \
            [2, ] * (tail_entity_end_pos - tail_entity_start_pos)


        # Account for [CLS] and [SEP]
        special_tokens_count = 2

        if head_entity_end_pos > max_seq_len - special_tokens_count - 1:
            continue
        if tail_entity_end_pos > max_seq_len - special_tokens_count - 1:
            continue

        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            rel_position_ids = rel_position_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        rel_position_ids += [0]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        rel_position_ids = [0] + rel_position_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        rel_position_ids = rel_position_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(rel_position_ids) == max_seq_len, "Error with rel_position_ids length {} vs {}".format(len(rel_position_ids), max_seq_len)

        head_entity_pos = [head_entity_start_pos, head_entity_end_pos]
        tail_entity_pos = [tail_entity_start_pos, tail_entity_end_pos]


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("words: %s" % " ".join([str(x) for x in example.words]))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("rel_position_ids: %s" % " ".join([str(x) for x in rel_position_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("re_label_id:  %d" % (re_label_id))
            logger.info("head_entity_pos:  (%d, %d)" % (head_entity_pos[0], head_entity_pos[1]))
            logger.info("tail_entity_pos:  (%d, %d)" % (tail_entity_pos[0], tail_entity_pos[1]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          rel_position_ids=rel_position_ids,
                          re_label_id=re_label_id,
                          head_entity_pos=head_entity_pos,
                          tail_entity_pos=tail_entity_pos,
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(
            examples, args.max_seq_len, tokenizer,
            span_identification_method=args.span_identification_method,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_rel_position_ids = torch.tensor([f.rel_position_ids for f in features], dtype=torch.long)
    all_re_label_ids = torch.tensor([f.re_label_id for f in features], dtype=torch.long)
    all_head_entity_pos = torch.tensor([f.head_entity_pos for f in features], dtype=torch.long)
    all_tail_entity_pos = torch.tensor([f.tail_entity_pos for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids,
                            all_attention_mask,
                            all_token_type_ids,
                            all_rel_position_ids,
                            all_re_label_ids,
                            all_head_entity_pos,
                            all_tail_entity_pos,
                            )
    return dataset
