# Knowledge Graph Attention Network
This is PyTorch & DGL implementation for the paper [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854). The code is modified from [KGAT-pytorch](https://gitee.com/hcxy0729/KGAT-pytorch#https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1905.07854).

## Introduction
Knowledge Graph Attention Network (KGAT) is a new recommendation framework tailored to knowledge-aware personalized recommendation. Built upon the graph neural network framework, KGAT explicitly models the high-order relations in collaborative knowledge graph to provide better recommendation with item side information.


## Environment
PyTorch  1.10.0
Python  3.8(ubuntu20.04)
CUDA  11.3
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
numpy pandas scipy scikit-learn tqdm

local:
PyTorch  2.5.1
Python  3.9
pip install dgl==1.0.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
numpy pandas scipy scikit-learn tqdm


## Run the Codes
KGAT amazon-book
```
python main_kgat.py --data_name amazon-book (default: --stopping_steps 10)
# local:
python main_kgat.py --data_name amazon-book --cf_batch_size 128 --kg_batch_size 128 --test_batch_size 256 --n_epoch 100
```

## Results
With my code, following are the results of each model when training with dataset `amazon-book`.

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| FM    | sample 1000 test users | 65         | 0.014400000683963299 | 0.14490722119808197 | 0.07221827559341328 |
| NFM   | sample 1000 test users | 56         | 0.013850000686943531 | 0.13833996653556824 | 0.0724611583347469  |
| BPRMF | all test users         | 65         | 0.014154779163154574 | 0.13356850621872207 | 0.06943918307731874 |
| ECFKG | all test users         | 41         | 0.013035656309061863 | 0.12247500353257905 | 0.06115661206228789 |
| CKE   | all test users         | 52         | 0.014507515353912879 | 0.13836056015380443 | 0.07225836488142431 |
| KGAT  | all test users         | 31         | 0.014817044902584718 | 0.14117674635791852 | 0.07526633940808744 |

Final results on all test users

| Model | Precision@20 | Recall@20 | NDCG@20 |
| :---: | :---:        | :---:     | :---:   |
| FM    | 0.0138       | 0.1309    | 0.0676  |
| NFM   | 0.0131       | 0.1246    | 0.0655  |
| BPRMF | 0.0142       | 0.1336    | 0.0694  |
| ECFKG | 0.0130       | 0.1225    | 0.0612  |
| CKE   | 0.0145       | 0.1384    | 0.0723  |
| KGAT  | 0.0148       | 0.1412    | 0.0753  |

        


