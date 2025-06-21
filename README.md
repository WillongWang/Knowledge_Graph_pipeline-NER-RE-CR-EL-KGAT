# Knowledge_Graph_pipeline-NER-RE-CR-EL-KGAT

This project includes the construction process and core technologies of a Knowledge Graph: Named Entity Recognition, Relation Extraction and Coreference Resolution, Entity Linking, and the PyTorch implementation of Knowledge Graph Attention Network for Recommendation.  

Testing Environment: PyTorch 1.10.0, Python 3.8 (Ubuntu 20.04), CUDA 11.3  
`pip install numpy pandas scipy scikit-learn tqdm`



## Named Entity Recognition   
In `/datasets`:
`conll03`'s each dictionary item's span_position contains the start_position and end_position of the entity in the context. The label files include `bio_labels.txt` (9 labels), `crf_bio_labels.txt` (CRF), `biaffine_labels.txt`; "impossible" indicates whether an entity is present in the context.  
`kgclue`'s `qa_data` (from [KgCLUE](https://github.com/CLUEbenchmark/KgCLUE)) converted to NER data (via NERdata.py) becomes `ner_data`; in `qa_data`, each line in `train.json` contains an answer in the format: entity ||| relation ||| entity, where all elements appear in the corresponding question (in Chinese); `ner_data` also contains three txt files.


### BERT subword tokenizer three approaches  
    # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, O, O; (sub_token_label_scheme == "v2")  
    # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, PAD, PAD; (sub_token_label_scheme == "v3", loss ignore [PAD])  
    # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, I-PER, I-PER; (sub_token_label_scheme == "v1")  


### How to Run  
Three models are used: BERT+MLP, BERT+CRF (Conditional random fields), BERT+Biaffine (the latter two use [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main), which should be downloaded to `/resources/chinese_bert_wwm_ext`)  
The `pad_token_label_id` (label ids for `[PAD]`) is set to `--ignore_index`, and `--is_flat_ner` is used (no Nested NER).  
```
pip install filelock regex seqeval pytorch-crf transformers osa
cd Named_Entity_Recognition
```

#### BERT+MLP  
```
./src/MLP/scripts/train_conll03_label_scheme_v1.sh
```
Check errors/logs with:  
```
cat experiments/MLP/logs/bert_MLP_conll03_101001.log
```  
The input consists of tokenized input's input_ids and label ids (`[PAD]` defaults to -100), producing logits for each token over 9 label classes to compute the loss.  
The same process applies for v2 and v3.

#### BERT+CRF  
```
./src/CRF/scripts/train_kgclue.sh
```
`pad_token_label_id=0`, and by default, it uses v3.  
The loss is computed as follows, adding the negative log-likelihood of a sequence of tags given some emission scores from the Linear-chain CRF:  
![](https://github.com/WillongWang/Knowledge_Graph_pipeline-NER-RE-CR-EL-KGAT/blob/main/g.png)  

$$
\boxed{- \log P(y_1, \dots, y_n \mid x) = - \left( h(y_1; x) + \sum_{k=1}^{n-1} g(y_k, y_{k+1}) + h(y_{k+1}; x) \right) + \log Z(x)}
$$  

Where $Z(x)$ is the normalizer.

In the package `torchcrf`, pseudocode for `_compute_normalizer`:  
```
# shape of logits (emissions): (batch_size (b), seq_length (l), num_tags (t)).(torch.)transpose(0,1)
for i in range(1,l):
 s = ( logits[0] + start_transitions (shape t) ).unsqueeze(2) + transitions (shape (t,t)) + logits[i].unsqueeze(1)
logsumexp( (logsumexp(s,dim=1) (shape (b,t)) + end_transitions (shape t) ),dim=1)
```  
This differs from the original equation. Let the normalizer at time step t be denoted as $Z_t$, and divide it into k components (corresponding to the number of label classes):

$$
\boxed{Z_t = Z_t^{(1)} + Z_t^{(2)} + \dots + Z_t^{(k)}}
$$

$$
\boxed{\mathbf{Z_{t+1}} = \begin{pmatrix} Z_{t+1}^{(1)} \\ \vdots \\ Z_{t+1}^{(k)} \end{pmatrix}, 
H(y_{t+1} | x) = \begin{pmatrix} e^{h_{t+1}(1 \mid x)} \\ \vdots \\ e^{h_{t+1}(k \mid x)} \end{pmatrix}, 
G matrix: G_{ij} = e^{g(y_i,y_j)}}
$$

$$
\boxed{
Z_{t+1} = Z_t G \otimes H(y_{t+1} \mid x)}
$$

`_viterbi_decode` implements the Viterbi algorithm, involving dynamic programming, ensuring global optimality.

#### BERT+Biaffine
```
./src/Biaffine/scripts/train_kgclue.sh
cat experiments/biaffine/logs/bert_biaffine_kgclue_110901.log
```  
`pad_token_label_id=0`, `tmp_label` is represented by the Biaffine matrix: ![](https://github.com/WillongWang/Knowledge_Graph_pipeline-NER-RE-CR-EL-KGAT/blob/main/gg.png)  
This 2D space represents entities, where the first dimension represents the start token s, and the second dimension represents the end token e. A tuple (s, e) corresponds to a span, and the values in the tuple represent the label.  
The `label_ids_biaffine` is computed as:  
```
label_sparse = scipy.sparse.coo_matrix(tmp_label, dtype=np.int_)
values = label_sparse.data
indices = np.vstack((label_sparse.row, label_sparse.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = label_sparse.shape
return label_sparse = torch.sparse.LongTensor(i, v, torch.Size(shape))
```  
For each span, logits are computed across two label classes (shape: (batch_size, sequence_length, sequence_length, num_labels)), followed by random sampling of logits and loss computation.

#### Test Results
F1 scores are in `results/conll03/results_1224.md`, `results/kgclue/results_1224.md`:  
| Model        | F1 Score |
|--------------|----------|
| MLP + v1     | 0.9063   |
| Biaffine     | 0.9845   |
| CRF          | 0.9895   |



## Relation Extraction
The `data/semeval10` dataset is derived by [OpenNRE](https://github.com/thunlp/OpenNRE), each data sample is a dictionary containing:
- `token`: tokenized sentence text  
- `h` (head entity) and `t` (tail entity): each with `name` and `position`  
- `relation`: one of 19 predefined classes (`semeval_rel2id.json`)  
The model outputs 19 logits corresponding to the 19 relation classes, and computes the loss based on the ground truth label.

Coreference Resolution can be modeled similarly as a relation extraction task.  
Model architecture depicted in [Matching the Blanks: Distributional Similarity for Relation Learning](https://aclanthology.org/P19-1279.pdf):
![ggg.png](https://github.com/WillongWang/Knowledge_Graph_pipeline-NER-RE-CR-EL-KGAT/blob/main/ggg.png)  
The code implementation via `--span_identification_method`:
- `v1`: No modification
- `v2`: Add entity markers  
- `v3`: Replace with entity markers

Specifications in `data_loader.py`'s function `convert_examples_to_features`, v2:  
```
head_entity_start_token="[unused0]", # BERT dictionary
head_entity_end_token="[unused2]",
tail_entity_start_token="[unused1]",
tail_entity_end_token="[unused3]",
# ... head_entity_start_token(head_entity_start_pos) head_entity head_entity_end_token(head_entity_end_pos) ...
# ... tail_entity_start_token(tail_entity_start_pos) tail_entity tail_entity_end_token(tail_entity_end_pos) ...
```  
v3, replace entity with special tokens:  
```
head_entity_token="[unused4]"
tail_entity_token="[unused5]"
```


### How to Run
```
cd Relation_Extraction_and_Coreference_Resolution
```

--use_entity_vector only:  
```
CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --overwrite_cache > experiments/logs/rebert_0.log &
```  
result: re_acc = 0.8378482972136223

--use_entity_vector + --include_nli_ops:  
```
CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --overwrite_cache --include_nli_ops > experiments/logs/rebert_0_nli.log &  
```
result: re_acc = 0.8471362229102167

--use_cls_vector & --use_entity_vector:  
```
CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_2 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_cls_vector --use_entity_vector  --overwrite_cache  > experiments/logs/rebert_2.log &
```  
result: re_acc = 0.8490712074303406

Feature construction in `model/modeling_bert.py`'s function `forward`:  
```
if self.args.use_cls_vector and self.args.use_entity_vector:
    features = torch.cat([pooled_output, head_hidden, tail_hidden], 1)
elif not self.args.use_cls_vector and self.args.use_entity_vector:
    if self.args.include_nli_ops:
     features = torch.cat([head_hidden, tail_hidden, head_hidden*tail_hidden, head_hidden-tail_hidden], 1)
    else:
     features = torch.cat([head_hidden, tail_hidden], 1)
elif self.args.use_cls_vector and not self.args.use_entity_vector:
    features = pooled_output
```  
`--include_nli_ops` is inspired by Natural Language Inference (NLI) design. For the BERT representations of text_a and text_b (denoted as vectors a and b), the combined feature is constructed as [a, b, a * b, a - b].



## Entity Linking  
This module performs alias matching and name disambiguation. The model is also based on [chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext/tree/main), which should be downloaded to `Entity_Linking/lesson6/resources/chinese_bert_wwm_ext`.

CHIP-CDN dataset is from [CBLUE](https://tianchi.aliyun.com/dataset/95414), refer to `Entity_Linking/lesson6/datasets/CBLUE_datasets/CHIP-CDN/CHIP-CDN/README.txt` for details. For`Entity_Linking/lesson6/datasets/CBLUE_datasets/CHIP-CDN/CHIP-CDN/CHIP-CDN_train.json`, each entry contains a `text` (can have multiple words) and corresponding standardized medical term (entity).

`cd Entity_Linking/lesson6`, first run the three files in `/data_process` to construct the dataset `datasets/CBLUE_datasets/CHIP-CDN/training_data_0120`, where Elasticsearch is used to recall entities that match the "text" but are false positives, and constructs negative samples accordingly.  
However, local setup in `Entity_Linking/lesson6/elasticSearch/安装与运行.md` has bugs; Instead, use the cloud version: log in to the official Elasticsearch website, create a new index with `index_name = "chip-cdn-0120"`, and replace the original local connection code `Elasticsearch("http://localhost:9200")` with the provided remote connection `Elasticsearch("...", api_key="...")`.  
These processes generate the dataset files `Entity_Linking/lesson6/datasets/CBLUE_datasets/CHIP-CDN/training_data_0120/train.txt` and `dev.txt`, where each line is labeled with 0 for negative samples and 1 for positive samples.


### How to Run
The code is modified from [DynaBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/DynaBERT)

**Step 1**  
Fine-tune the pretrained BERT model (BertForSequenceClassification)
```
python DynaBERT/run_glue.py --seed 11901 --model_type bert --task_name cdn --do_train --data_dir datasets/CBLUE_datasets/CHIP-CDN/training_data_0120 --model_dir resources/chinese_bert_wwm_ext --output_dir experiments/outputs/bert_cdn_11901_finetuning --max_seq_length 48 --learning_rate 2e-5 --warmup_steps 300 --per_gpu_train_batch_size 256 --per_gpu_eval_batch_size 256 --num_train_epochs 5 --training_phase finetuning --evaluate_during_training True --logging_steps 200
```  
The model input consists of input_ids for two sentences per line. The `[CLS]` token representation is processed by the pooler and classifier to produce two logits, which are then used to compute the loss.

**Step 2**  
Test on the CHIP-CDN dev set
```
python eval_el.py --model_type bert --task_name cdn --data_dir datasets/CBLUE_datasets/CHIP-CDN/CHIP-CDN/CHIP-CDN_dev.json --max_seq_length 48 --per_gpu_eval_batch_size 128 --model_dir experiments/outputs/bert_cdn_11901_finetuning --output_dir experiments/outputs/bert_cdn_11901_finetuning/dev_predicted_results.txt
```  
First, Elasticsearch recalls the `topk` (default 128) entities matching the "text" as candidates. Each candidate entity is paired with the text and fed into the model. If the logit for label 1 exceeds a given threshold (0.9, 0.95, 0.98, 0.99, 0.995), the `entity_name` is appended to the predictions. Precision, recall, and F1 scores are then calculated.

Evaluation results are saved in `experiments/outputs/bert_cdn_11901_finetuning` and the scores are not high.



## KGAT-pytorch
This is PyTorch & DGL implementation for the paper [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854). The code is modified from [KGAT-pytorch](https://gitee.com/hcxy0729/KGAT-pytorch#https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1905.07854).
In `datasets/amazon-book`, `kg_final.txt` contains Knowledge Graph triples per line with format (t (top), r, h (head)); `train.txt` and `test.txt` contain lines representing user ID paired with the product IDs they purchased.

For KG embedding, the TransR model is used:  

$$
\boxed{g(h, r, t) = \| W_r e_h + e_r - W_r e_t \|^2}
$$

Requirements:  
```
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
# or in local environment: PyTorch 2.5.1, Python 3.9; comment out loader_kgat.py line 169: g.readonly()
pip install dgl==1.0.2+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
```

### How to Run  
default early stopping: --stopping_steps 10  
```
python main_kgat.py --data_name amazon-book
# local:
python main_kgat.py --data_name amazon-book --cf_batch_size 128 --kg_batch_size 128 --test_batch_size 256 --n_epoch 100
```

### Results  
early stopping at 41 epochs  
| epoch_idx | precision@20           | recall@20             | ndcg@20              |
|-----------|-----------------------|----------------------|----------------------|
| 41.0      | 0.014543638940191200  | 0.13864989744457800  | 0.07386811813034540  |
