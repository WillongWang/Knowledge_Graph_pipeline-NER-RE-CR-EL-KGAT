# coding=utf-8

import csv
import json
import torch

import numpy as np


def transform_label_matrix2spans(label_matrix, id2label_map=None):
    # Rm为l*l*c的tensor,表征着每一个起始i终止i的片段属于各个实体的概率
    # 将该矩阵解码为实体列表，每一个实体用tuple表示，（category_id, pos_b, pos_e）
    # 如果label标签为True，则输入为l*l的tensor
    entities = []
    cate_tensor = label_matrix
    cate_indices = torch.nonzero(cate_tensor)
    for index in cate_indices:
        cate_id = int(cate_tensor[index[0], index[1]])
        label_name = id2label_map[cate_id]
        entities.append((int(index[0]), int(index[1]), label_name))

    return entities


def Rm2entities(Rm, is_flat_ner=True, id2label_map=None):
    Rm = Rm.squeeze(0)

    # get score and pred l*l
    score, cate_pred = Rm.max(dim=-1)

    # fliter mask
    # mask category of none-entity
    seq_len = cate_pred.shape[1]
    zero_mask = (cate_pred == torch.tensor([0]).float().to(score.device))
    score = torch.where(zero_mask.byte(), torch.zeros_like(score), score)
    # pos_b <= pos_e check
    score = torch.triu(score)  # 上三角函数
    cate_pred = torch.triu(cate_pred)

    # get all entity list
    all_entity = []
    score_shape = score.shape
    score = score.reshape(-1)
    cate_pred = cate_pred.reshape(-1)
    entity_indices = (score != 0).nonzero(as_tuple = False).squeeze(-1)
    for i in entity_indices:
        i = int(i)
        score_i = float(score[i].item())
        cate_pred_i = int(cate_pred[i].item())
        pos_s = i // seq_len
        pos_e = i % seq_len
        all_entity.append((pos_s,pos_e,cate_pred_i,score_i))

    # sort by score
    all_entity = sorted(all_entity, key=lambda x:x[-1])
    res_entity = []

    for ns, ne, t, _ in all_entity:
        for ts, te, _ in res_entity:
            if ns < ts <= ne < te or ts < ns <= te < ne:
                # for both nested and flat ner no clash is allowed
                break

            if is_flat_ner and (ns <= ts <= te <= ne or ts <= ns <= ne <= te):
                # for flat ner nested mentions are allowed
                break

        else:
            res_entity.append((ns, ne, id2label_map[t]))
    return set(res_entity)


def count_same_entities(label_items, pred_items):
    count = 0
    for item in label_items:
        if item in pred_items:
            count += 1
    return count


def trans_span2bio(
    input_seq,
    max_seq_len=None,
    real_seq_len=None,
    list_entities=None,
):
    # 将 span 转化为 BIO 序列
    list_labels = ["O"] * max_seq_len
    # print(list_labels)
    # print("real_seq_len: ", real_seq_len)

    for s, e, lab in list_entities:
        list_labels[s] = "B-" + lab
        list_labels[s + 1 : e + 1] = ["I-" + lab] * (e - s)

    list_labels = list_labels[: int(real_seq_len)]
    return list_labels


def get_biaffine_table(Rm, list_spans):
    Rm = Rm.squeeze(0)

    score, cate_pred = Rm.max(dim=-1)

    # fliter mask
    # mask category of none-entity
    seq_len = cate_pred.shape[1]

    # tmp_label = np.zeros((seq_len, seq_len))
    biaffine_label = np.full((seq_len, seq_len), 'O', dtype=np.str)
    for span in list_spans:
        biaffine_label[span[0], span[1]] = span[2]

    return biaffine_label
