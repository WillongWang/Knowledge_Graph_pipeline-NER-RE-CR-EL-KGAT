
import csv
import json
import torch
from seqeval.metrics import precision_score, recall_score, f1_score  # 专门用于序列标注评分用的
import numpy as np

def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


def get_sentence_frame_acc(slot_preds, slot_labels):
    """For the cases that all the slots are correct (in one sentence)"""

    # Get the slot comparision result
    slot_result = []    # 一整句话全部预测正确的就为True
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = slot_result.mean()
    return {
        "sementic_frame_acc": sementic_acc
    }


def compute_metrics(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(slot_preds, slot_labels)

    results.update(slot_result)
    results.update(sementic_result)

    return results