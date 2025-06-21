# coding=utf-8
# 2020.08.28 - Changed regular evaluation to evaluation with adaptive width and depth
#              Huawei Technologies Co., Ltd <houlu3@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import numpy as np
import torch
from elasticsearch import Elasticsearch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm

import sys
sys.path.insert(0, "./")

from DynaBERT.transformers import (BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          )

from DynaBERT.transformers import glue_compute_metrics as compute_metrics
from DynaBERT.transformers import glue_output_modes as output_modes
from DynaBERT.transformers import glue_processors as processors
from DynaBERT.transformers import glue_convert_examples_to_features as convert_examples_to_features

from DynaBERT.transformers import InputExample

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_length=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.seq_length = seq_length
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features_test(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        try:
            if output_mode == "classification":
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
        except:
            label_id = 0

        # if ex_index < 1:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: {}".format(example.label))
        #     logger.info("label_id: {}".format(label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          seq_length=seq_length))
    return features


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def retrieve_by_es(entity_mention, es_conn, index_name="chip-cdn-0120", topk=200):
    must_conditions = [
        {
            "match": {
                "entity_name": entity_mention,
            }
        }
    ]
    doc = dict()
    doc["query"] = {
        "bool": {
            "must": must_conditions,
            "must_not": [],
        }
    }
    doc["size"] = topk

    list_retrieved_results = []
    try:
        results = es_conn.search(index=index_name, body=doc)
        # print(results['hits']['hits'])

        for res in results['hits']['hits']:
            list_retrieved_results.append(res["_source"])

    except Exception as e:
        print(e)

    return list_retrieved_results


def get_tensor_data(features):
    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_seq_lengths)
    return tensor_data


def inference_once(input_text,
                   args, model,
                   tokenizer, es_conn=None,
                   index_name=None,
                   topk=128,
                   threshold=0.98
                   ):
    processor = processors[args.task_name]()
    output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    if args.task_name in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
        label_list[1], label_list[2] = label_list[2], label_list[1]

    # ES召回
    retrieved_samples = retrieve_by_es(
        input_text,
        es_conn,
        index_name=index_name,
        topk=topk,
    )

    # 召回的candidte与mention形成 匹配任务的 examples
    examples = []
    for i, samp in enumerate(retrieved_samples):
        text_a = input_text
        text_b = samp["entity_name"]

        guid = str(i)
        examples.append(
            InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label="0")
        )

    # examples特征化
    features = convert_examples_to_features_test(
        examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        output_mode
    )

    # tensor 化
    eval_dataset = get_tensor_data(features)

    # 模型预测
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            # the following print function is to output original sentence for the visualization, wrt. batch_size=1
            # print(tokenizer.decode(inputs['input_ids'][0].cpu().numpy()))
            outputs = model(**inputs)
            logits = outputs[0]

        nb_eval_steps += 1
        if preds is None:
            preds = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.softmax(logits, dim=-1).detach().cpu().numpy(), axis=0)

    # 得到预测结果
    # preds = np.argmax(preds, axis=1).tolist()
    scores = preds[:, 1].tolist()
    # print(scores)

    # 预测匹配的实体名称为
    matched_entities = []
    for score, samp in zip(scores, retrieved_samples):
        if score > threshold:
            matched_entities.append(samp["entity_name"])

    return matched_entities


def eval_all(data_dir,
             args,
             model,
             tokenizer,
             es_conn=None,
             index_name=None,
             topk=128,
             threshold=0.97):
    list_samples = json.load(
        open(data_dir, "r", encoding="utf-8")
    )

    output_dir = args.output_dir
    f_out = open(output_dir, "w", encoding="utf-8")

    nb_correct = 0
    nb_truth = 0
    nb_pred = 0
    for samp in tqdm(list_samples):
        mention = samp["text"]
        normalized_result = samp["normalized_result"]
        normalized_result = normalized_result.split("##")
        normalized_result = [tokenizer.basic_tokenizer._clean_text(w.strip()) for w in normalized_result]
        normalized_result = [w for w in normalized_result if w]

        # 进行一次预测
        matched_entities = inference_once(
            mention,
            args,
            model,
            tokenizer,
            es_conn=es_conn,
            index_name=index_name,
            topk=topk,
            threshold=threshold,
        )

        nb_correct += len([w for w in matched_entities if w in normalized_result])
        nb_pred += len(matched_entities)
        nb_truth += len(normalized_result)

        print(nb_correct, nb_pred, nb_truth)

        tmp1 = "##".join(matched_entities)
        tmp2 = "##".join(normalized_result)
        f_out.write(
            f"{mention}\t{tmp2}\t{tmp1}\n"
        )

    precision = nb_correct / nb_pred if nb_pred > 0 else 0
    recall = nb_correct / nb_truth if nb_truth > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    f_out.write(
        "\n" * 5
    )
    f_out.write(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    f_out.write(
        "\n" * 5
    )
    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--model_dir", type=str,
                        help="The teacher model dir.")
    parser.add_argument('--depth_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")
    parser.add_argument('--width_mult', type=str, default='1.',
                        help="the possible depths used for training, e.g., '1.' is for default")

    args = parser.parse_args()
    # args.model_dir = os.path.join(args.model_dir, 'best')
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(args.model_dir, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_dir, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_dir, config=config)
    model.to(args.device)
    # model.apply(lambda m: setattr(m, 'depth_mult', float(args.depth_mult)))
    # model.apply(lambda m: setattr(m, 'width_mult', float(args.width_mult)))

    # 在cdn dev级上进行评测
    es_config = {
        "ip": "localhost",
        "port": 9200,
    }
    # es_conn = Elasticsearch(
    #     [es_config["ip"]],
    #     http_auth=None,
    #     port=es_config["port"],
    # )
    es_conn = Elasticsearch("http://localhost:9200")
    print(es_conn.indices)
    index_name = "chip-cdn-0120"

    for thres in [0.9, 0.95, 0.96, 0.98, 0.99, 0.995]:
        args.output_dir = args.output_dir[: -4] + f"_threshold_{thres}" + ".txt"

        results = eval_all(
            args.data_dir,
            args,
            model,
            tokenizer,
            es_conn=es_conn,
            index_name=index_name,
            threshold=thres
        )
        print(results)

if __name__ == "__main__":
    main()
