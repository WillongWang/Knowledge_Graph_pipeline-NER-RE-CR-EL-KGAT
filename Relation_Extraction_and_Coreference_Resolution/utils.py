import json
import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import ReBERT

MODEL_CLASSES = {
    'bert': (BertConfig, ReBERT, BertTokenizer),
}


def get_re_labels(args):

    re_label2id = json.load(
        open(
            os.path.join(args.data_dir, args.task, args.re_label_file),
            'r',
            encoding='utf-8',
        )
    )

    return re_label2id


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(re_preds, re_labels):
    assert len(re_preds) == len(re_labels)
    results = {}
    re_result = get_re_acc(re_preds, re_labels)

    results.update(re_result)

    return results


def get_re_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "re_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]
