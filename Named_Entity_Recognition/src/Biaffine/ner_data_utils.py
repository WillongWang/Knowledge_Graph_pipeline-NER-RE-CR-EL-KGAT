# coding=utf-8

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

    def __init__(self, input_ids, input_mask, segment_ids, label_ids_biaffine):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.label_ids = label_ids
        self.label_ids_biaffine = label_ids_biaffine


def read_example_form_CCKS(data_dir, mode):
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


def convert_MRC_examples_to_features(examples, label_list, max_seq_length, tokenizer, cls_token_at_end=False,
                                     cls_token="[CLS]",
                                     cls_token_segment_id=1, sep_token="[SEP]", sep_token_extra=False,
                                     pad_on_left=False, pad_token=0,
                                     pad_token_segment_id=0, pad_token_label_id=0, sequence_a_segment_id=0,
                                     mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    # print(pad_token_label_id)
    features = []
    max_seq_length_real = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        count = 0

        label_seqs = []
        for label in example.labels:
            temp = []
            temp.append(label[0])
            temp.append(label[1])
            temp.append(label[2])
            label_seqs.append(temp)

        # label_seqs = [label.copy() for label in example.labels]

        for num, word in enumerate(example.words):
            # print(word, num)
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                raise ValueError("{} annotation error!".format(word))
            tokens.extend(word_tokens)

            count += len(word_tokens)

            for id, label in enumerate(example.labels):
                label_oringin_start = label[0]
                label_oringin_end = label[1]

                if num < label_oringin_start:
                    label_seqs[id][0] += (len(word_tokens) - 1)
                    label_seqs[id][1] += (len(word_tokens) - 1)
                    continue
                elif num > label_oringin_end:
                    continue
                else:
                    label_seqs[id][1] += (len(word_tokens) - 1)
        #
        # print("oringin_labels: ", example.labels)
        # print("tokens: ", tokens)
        # print(len(example.words))
        # print(example.words)
        # print("label_seqs: ", label_seqs)
        # Account for [CLS] and [SEP] with "- 2" and with "-3" for RoBERTa.
        #
        special_tokens_count = 3 if sep_token_extra else 2

        label_add_list = []
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]

            for id, label in enumerate(label_seqs):
                label_add = []
                label_oringin_start = label[0]
                label_oringin_end = label[1]

                if max_seq_length - special_tokens_count <= label_oringin_start:  # 删除
                    continue
                elif max_seq_length - special_tokens_count == label_oringin_start + 1:  # 边界值
                    label_add.append(max_seq_length - special_tokens_count - 1)
                    label_add.append(max_seq_length - special_tokens_count - 1)
                    label_add.append(label[2])
                    label_add_list.append(label_add)
                elif label_oringin_end + 1 <= max_seq_length - special_tokens_count:
                    label_add_list.append(label)
                else:
                    label_add.append(label[0])
                    label_add.append(max_seq_length - special_tokens_count - 1)
                    label_add.append(label[2])
                    label_add_list.append(label_add)
        else:
            label_add_list = [label.copy() for label in label_seqs]
        # print(tokens)
        # print(len(tokens))
        # print(label_add_list)  # 截断之后的label,label_seqs变为label_add_list
        #
        tokens += [sep_token]
        # label_seqs += ["O"]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            # label_seqs += ["O"]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            # label_seqs += ["O"]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            # label_seqs = ["O"] + label_seqs  ###
            segment_ids = [cls_token_segment_id] + segment_ids
            for ele in label_add_list:
                ele[0] += 1
                ele[1] += 1

        # print("tokens: ", tokens)
        # print("label_seqs: ", label_seqs)
        # print("label_seqs: ", label_seqs)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and O for padding tokens. Only real
        # tokens are attended to
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            # label_seqs = (["O"] * padding_length) + label_seqs   ###
            for ele in label_add_list:
                ele[0] += padding_length
                ele[1] += padding_length
        else:
            input_ids += ([pad_token] * padding_length)
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([pad_token_segment_id] * padding_length)
            # label_seqs += (["O"] * padding_length)  ###

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(label_seqs) == max_seq_length  ###

        # biaffine NER labels
        '''
        labels: [0, 1, 1, 1, 1, 2, 2, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
        '''
        # label_add_list: [[8, 9, 'GPE'], [1, 6, 'ORG']]
        label_ids_biaffine = trans_label_MRC_EN(label_add_list, max_seq_length, label_map)  ###
        # print('label_ids_biaffine:', label_ids_biaffine)

        # log
        if ex_index < 3:
            print("###############")
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_seqs: %s", json.dumps(label_seqs, ensure_ascii=False))
            # logger.info("label_ids_biaffine: %s", json.dumps(label_ids biaffine))
            print("###############")

        # package data
        if len(input_ids) == max_seq_length:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids_biaffine=label_ids_biaffine,
                )
            )
    return features


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


ner_processors = {
    "ace2004": MRCProcessor,
    "ace2005": MRCProcessor,
    "conll03": MRCProcessor,
    "genia": geniaProcessor,
    "zh_msra": MRCProcessor,
    "zh_onto4": MRCProcessor,
}
