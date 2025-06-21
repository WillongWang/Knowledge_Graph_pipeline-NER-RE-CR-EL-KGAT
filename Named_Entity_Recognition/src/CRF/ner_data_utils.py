import torch
import logging
import os
import copy
import json

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Construct a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            apecified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

    def __repr__(self):
        # print(类的实例)：__repr__ 规定的内容
        return json.dumps(self.__dict__, ensure_ascii=False)

class InputFeatures(object):
    """A single set of features of data"""

    def __init__(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        # self.label_ids = label_ids
        self.slot_labels_ids = slot_labels_ids


def trans_label_MRC_EN(label_seq, matrix_lenth, label_map):  # [[8, 9, 'GPE'], [1, 6, 'ORG']]
    import numpy as np
    from scipy.sparse import coo_matrix
    import torch

    # conll 2 doccano json
    tmp_tokens = ["ant"] * matrix_lenth
    # print('tmp_tokens：', tmp_tokens)
    # print('label_seq:', label_seq)
    # _, list_spans = conll2doccano_json(tmp_tokens, label_seq)  # [[0, 1, '胸', '部位'], [2, 4, '腹部', '部位']]
    # print(list_spans)

    list_spans = [(span[0], span[1], span[2]) for span in label_seq]
    # print(list_spans)

    tmp_label = np.zeros((matrix_lenth, matrix_lenth))
    # print("tmp_label: ", tmp_label)

    for span in list_spans:
        tmp_label[span[0], span[1]] = label_map[span[2]]

    # print("tmp_label: ", tmp_label)

    label_sparse = coo_matrix(tmp_label, dtype=np.int)
    # print("label_sparse: ", label_sparse)

    values = label_sparse.data
    # print("values: ", values)

    # print("label_sparse.row: ", label_sparse.row)
    # print("label_sparse.col: ", label_sparse.col)
    indices = np.vstack((label_sparse.row, label_sparse.col))
    # print("indices: ", indices)

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = label_sparse.shape
    label_sparse = torch.sparse.LongTensor(i, v, torch.Size(shape))

    return label_sparse

class MRCProcessor():
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self, path):
        """See base class."""
        if path.endswith("txt"):
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels

        if path.endswith("json"):
            labels = json.load(
                open(path, "r", encoding="utf-8")
            )
            return labels

        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", ]

    def _create_examples(self, data_dir, mode):
        """Creates examples for the training and dev sets."""
        # mrc格式
        file_path = os.path.join(data_dir, "mrc-ner.{}".format(mode))
        all_data = json.load(open(file_path, encoding="utf-8"))
        examples = []
        label_num = all_data[-1]["qas_id"].split(".")[-1]  # 就是标签的个数（不包括O）（从1开始）
        sample_num = all_data[-1]["qas_id"].split(".")[0]  # 就是样本的个数 （从0开始）
        label_num = int(label_num)

        words_list = []
        label_list = []
        guid_index = 0

        for i, line in enumerate(all_data):

            if not line:
                continue

            if i % label_num == 0:
                words_list = line["context"].split(" ")
                guid_index += 1

            if (i % label_num == (label_num - 1)) and (
            not line["start_position"]) and label_list == []:  # 去掉没有实体标签的样本数据
                guid_index -= 1
                continue

            if not line["start_position"]:
                if i != 0 and (i % label_num == (label_num - 1)):
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words_list,
                            labels=label_list)
                    )
                    label_list = []
                continue

            start_position_list = line["start_position"]
            end_positions_list = line["end_position"]
            for num, start in enumerate(start_position_list):
                list_spans = (start, end_positions_list[num], line["entity_label"])
                label_list.append(list_spans)

            # assert len(tokens) == len(ner_tags)
            if i != 0 and (i % label_num == (label_num - 1)):
                examples.append(
                    InputExample(
                        guid="{}-{}".format(mode, guid_index),
                        words=words_list,
                        labels=label_list)
                )
                label_list = []
        return examples


class geniaProcessor():
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self, path):
        """See base class."""
        if path.endswith("txt"):
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels

        if path.endswith("json"):
            labels = json.load(
                open(path, "r", encoding="utf-8")
            )
            return labels

        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", ]

    def _create_examples(self, data_dir, mode):
        """Creates examples for the training and dev sets."""
        # genia数据
        file_path = os.path.join(data_dir, "mrc-ner.{}".format(mode))
        all_data = json.load(open(file_path, encoding="utf-8"))
        examples = []

        label_num = 5

        words_list = []
        label_list = []
        guid_index = 0

        for i, line in enumerate(all_data):

            if not line:
                continue

            if i % label_num == 0:
                words_list = line["context"].split(" ")
                guid_index += 1

            if not line["start_position"]:
                if i != 0 and (i % label_num == (label_num - 1)):
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words_list,
                            labels=label_list)
                    )
                    label_list = []
                continue

            start_position_list = line["start_position"]
            end_positions_list = line["end_position"]
            for num, start in enumerate(start_position_list):
                list_spans = (start, end_positions_list[num], line["entity_label"])
                label_list.append(list_spans)

            # assert len(tokens) == len(ner_tags)
            if i != 0 and (i % label_num == (label_num - 1)):
                examples.append(
                    InputExample(
                        guid="{}-{}".format(mode, guid_index),
                        words=words_list,
                        labels=label_list)
                )
                label_list = []
        return examples


def read_example_from_CCKS(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.json".format(mode))
    all_data = json.load(open(file_path, encoding="utf-8"))

    examples = []

    for i, samp in enumerate(all_data):
        examples.append(
            InputExample(
                guid="{}-{}".format(mode, i),
                words=samp["sentence"],
                labels=samp["labeled entities"])
        )

    return examples


def read_example_form_MRC(data_dir, mode):
    # mrc格式
    file_path = os.path.join(data_dir, "mrc-ner.{}".format(mode))
    all_data = json.load(open(file_path, encoding="utf-8"))
    examples = []
    label_num = all_data[-1]["qas_id"].split(".")[-1] # 就是标签的个数（不包括O）（从1开始）
    sample_num = all_data[-1]["qas_id"].split(".")[0]  # 就是样本的个数 （从0开始）
    label_num = int(label_num)

    words_list = []
    label_list = []
    guid_index = 0

    for i, line in enumerate(all_data):

        if not line:
            continue

        if i % label_num == 0:
            words_list = line["context"].split(" ")
            guid_index += 1

        if (i % label_num == (label_num - 1)) and (not line["start_position"]) and label_list == []: # 去掉没有实体标签的样本数据
            guid_index -= 1
            continue

        if not line["start_position"]:
            if i != 0 and (i % label_num == (label_num - 1)):
                examples.append(
                    InputExample(
                        guid="{}-{}".format(mode, guid_index),
                        words=words_list,
                        labels=label_list)
                )
                label_list = []
            continue

        start_position_list = line["start_position"]
        end_positions_list = line["end_position"]
        for num, start in enumerate(start_position_list):
            list_spans = (start, end_positions_list[num], line["entity_label"])
            label_list.append(list_spans)

        # assert len(tokens) == len(ner_tags)
        if i != 0 and (i % label_num == (label_num - 1)) :
            examples.append(
                InputExample(
                    guid="{}-{}".format(mode, guid_index),
                    words=words_list,
                    labels=label_list)
            )
            label_list = []

    return examples


def convert_MRC_examples_to_featuresCRF(examples,
                                        label_list,
                                        max_seq_length,
                                        tokenizer,
                                        cls_token_at_end=False,
                                        cls_token="[CLS]",
                                        cls_token_segment_id=1,
                                        sep_token="[SEP]",
                                        sep_token_extra=False,
                                        pad_on_left=False,
                                        pad_token=0,
                                        pad_token_segment_id=0,
                                        pad_token_label_id=-100,
                                        sequence_a_segment_id=0,
                                        mask_padding_with_zero=True,
                                        sub_token_label_scheme="v1",
                                        ):

    label_map = {label: i for i, label in enumerate(label_list)}
    label_id2label = {i: label for i, label in enumerate(label_list)}

    # print(pad_token_label_id)
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    features = []
    max_seq_length_real = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 100 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        # 先在单词级，给出每个单词的label
        tokens = []
        slot_labels_ids = []
        ###
        labels_seq_list = ["O" for _ in range(len(example.words))]
        for label_span in example.labels:
            s = int(label_span[0])
            e = int(label_span[1])
            l = label_span[-1]
            labels_seq_list[s] = "B-" + l
            for _ in labels_seq_list[s + 1: e + 1]:
                labels_seq_list[s + 1: e + 1] = ["I-" + l for _ in range(e - s)]

        slot_labels = []  # 转换为ids的序列标签列表
        for s in labels_seq_list:
            slot_labels.append(
                label_map[s])  # change
            # print(s, label_map[s], label_map[s])

        # 数据中有一些 输入 不是单词级别，比如日期，需要做转化
        words_new = []
        slot_labels_new = []
        for word, slot_label in zip(example.words, slot_labels):
            words_ = tokenizer.basic_tokenizer.tokenize(word)
            if not words_:
                continue

            words_new.extend(words_)

            slot_label_nonstart = label_map[label_id2label[slot_label].replace("B-", "I-")]
            slot_labels_new.extend([int(slot_label)] + [slot_label_nonstart] * (len(words_) - 1))

        # 转为 sub-token 级别
        for word, slot_label in zip(words_new, slot_labels_new):
            word_tokens = tokenizer.tokenize(word)  # 分词  将词输入到bert的tokenizer中去将它转化为bert词表中的tokens  ['i']
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)

            # A-word : B-PER;
            # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, O, O; (sub_token_label_scheme == "v2")
            # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, PAD, PAD;   (sub_token_label_scheme == "v3")
            # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, I-PER, I-PER;  (sub_token_label_scheme == "v1")

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens

            if sub_token_label_scheme == "v1":
                slot_label_nonstart = label_map[label_id2label[slot_label].replace("B-", "I-")]
                slot_labels_ids.extend([int(slot_label)] + [slot_label_nonstart] * (len(word_tokens) - 1))
            elif sub_token_label_scheme == "v2":
                slot_label_o = label_map["O"]
                slot_labels_ids.extend([int(slot_label)] + [slot_label_o] * (len(word_tokens) - 1))
            else:
                slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

            # pad_token_label_id: -100, loss function  忽略的label编号 只关注first label的token是什么
            # 特殊token、padding、拆开的token都是给-100
            # Account for [CLS] and [SEP]

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_length - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]  # [SEP] label: pad_token_label_id
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids  # [CLS] label: pad_token_label_id
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)  # 长度补齐

        assert len(input_ids) == max_seq_length, "Error with input length {} vs {}".format(len(input_ids), max_seq_length)
        assert len(attention_mask) == max_seq_length, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_length)
        assert len(token_type_ids) == max_seq_length, "Error with token type length {} vs {}".format(len(token_type_ids),max_seq_length)
        assert len(slot_labels_ids) == max_seq_length, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_length)

        if ex_index < 10:  # 1198 < ex_index < 1200:
            print("*** Example ***")
            print("guid: %s" % example.guid)
            print("original words: %s" % " ".join([str(x) for x in example.words]))
            print("original slot_labels: %s" % " ".join([label_list[int(x)] for x in slot_labels]))
            print("tokens: %s" % " ".join([str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          slot_labels_ids=slot_labels_ids,
                          ))

    return features

def get_labels(path):

    print(path)

    if path.endswith("txt"):
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels

    if path.endswith("json"):
        labels = json.load(
            open(path, "r", encoding="utf-8")
        )
        return labels

    return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",]



ner_processors = {
    "ace2004": MRCProcessor,
    "ace2005":MRCProcessor,
    "conll03":MRCProcessor,
    "genia":geniaProcessor,
    "zh_msra":MRCProcessor,
    "zh_onto4":MRCProcessor,
}