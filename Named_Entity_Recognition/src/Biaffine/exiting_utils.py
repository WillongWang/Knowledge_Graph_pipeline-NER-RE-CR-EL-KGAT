# coding=utf-8
import Levenshtein
import torch
import numpy as np
from src.Biaffine.biaffine_utils import Rm2entities
from scipy.sparse import coo_matrix

import torch.nn.functional as F


def seq_edit(seq_a, seq_b):
    '''
    "hello"  "world" 长度可以不一致
    4 1
    '''
    dis = Levenshtein.distance(seq_a, seq_b)
    # print(dis)
    return dis


def seq_ham(seq_a, seq_b):  ###太严格不太好搞
    if seq_a != seq_b:
        print("seq should be equal!")
    '''
    "hello"  "world" 长度必须一致
    4 1
    '''
    return Levenshtein.hamming(seq_a, seq_b)


def seq_leven_ratio(seq_a, seq_b):
    '''
    "hello"  "world" 长度可以不一致
    o.2 0.666 0.444
    '''
    return Levenshtein.ratio(seq_a, seq_b)


def seq_jaro(seq_a, seq_b):
    '''
    "hello"  "world" 长度可以不一致
    o.2 0.666 0.444
    '''
    return Levenshtein.jaro(seq_a, seq_b)


def seq_jaro_winkler(seq_a, seq_b):
    '''
    "hello"  "world" 长度可以不一致
    o.2 0.666 0.444
    '''
    return Levenshtein.jaro_winkler(seq_a, seq_b)


def dot_product(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    """
    calculate representation dot production

    """
    tensor_1 = tensor_1.reshape(1, 1, -1, )
    tensor_2 = tensor_2.reshape(1, 1, -1, )

    return torch.bmm(tensor_1, torch.transpose(tensor_2, -1, -2)).squeeze()


def cosine_similarity(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    """
    calculate representation cosine similarity, note that this is different from torch version(that compute parwisely)

    """
    dot_prod = dot_product(tensor_1, tensor_2)

    tensor_1_norm = torch.norm(tensor_1, keepdim=False)
    tensor_2_norm = torch.norm(tensor_2, keepdim=False)

    sim = dot_prod / (tensor_1_norm * tensor_2_norm)
    return sim


def frobenius_distance(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    """
    calculate representation L2 similarity

    """

    dis = torch.norm(tensor_1 - tensor_2, keepdim=False, p="fro")

    tensor_1_norm = torch.norm(tensor_1, keepdim=False, p="fro")
    tensor_2_norm = torch.norm(tensor_2, keepdim=False, p="fro")

    sim = dis / max(tensor_1_norm, tensor_2_norm)

    return sim


def inf_distance(tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
    """
    calculate representation inf similarity

    """

    dis = torch.norm(tensor_1 - tensor_2, keepdim=False, p=float('inf'))

    tensor_1_norm = torch.norm(tensor_1, keepdim=False, p=float('inf'))
    tensor_2_norm = torch.norm(tensor_2, keepdim=False, p=float('inf'))

    sim = dis / max(tensor_1_norm, tensor_2_norm)

    return sim


def kl_distance(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), -1)

    return torch.mean(_kl)


def tensor2str(Rm, id2label_map=None):
    score, cate_pred = Rm.max(dim=-1)
    cate_pred = cate_pred.reshape(-1)
    label_data = [id2label_map[i] for i in cate_pred]
    string = ''.join(label_data)
    return string


### 去学习出来两者之间的关系

exit_processers = {
    "origin": None,
    "seq_edit": seq_edit,
    # "seq_ham": seq_ham,
    # "seq_leven_ratio": seq_leven_ratio,
    # "seq_jaro": seq_jaro,
    # "seq_jaro_winkler": seq_jaro_winkler,
    "dot_product": dot_product,
    "cosine_similarity": cosine_similarity,
    "frobenius_distance": frobenius_distance,
    "inf_distance": inf_distance,
    "kl_distance": kl_distance,
}


if __name__ == '__main__':
    a = seq_edit('hello', 'world')
    print(a)
