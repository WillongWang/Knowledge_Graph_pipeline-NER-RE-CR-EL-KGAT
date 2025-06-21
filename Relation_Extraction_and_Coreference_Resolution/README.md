## Model Architecture

- Predict `re label` from **one BERT model** 
- total_loss = re_loss 


Relation_Extraction_and_Coreference_Resolution# 

CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --overwrite_cache > experiments/logs/rebert_0.log &

re_acc = 0.8378482972136223

CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --overwrite_cache --include_nli_ops > experiments/logs/rebert_0_nli.log &

re_acc = 0.8471362229102167

CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_2 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_cls_vector --use_entity_vector  --overwrite_cache  > experiments/logs/rebert_2.log &

re_acc = 0.8490712074303406


## Dependencies
- transformers==3.0.2

## Dataset

与清华大学 OpenNRE 工具 中的划分是一致的；

|       | Train  | Dev | Test |  Slot Labels |
| ----- | ------ | --- | ---- |  ----------- |
| SemEval10  | 6507  | 1493 | 2717  |  19         |



- The number of labels are given by the dataset.



## Training & Evaluation

默认设定是 上节课的 第六个模型: 句子中加入entity marker;实体的向量表征拼接;


```bash

# do not use cls_vector, use entity_vector
CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_0 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --overwrite_cache > experiments/logs/rebert_0.log &

# on test
re_acc = 0.8378482972136223

```


### ablation on features for classification

```bash

# For SemEval10

# features for classification

#  use cls_vector, not to use entity_vector
CUDA_VISIBLE_DEVICES="0" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_1 --do_train --do_eval --train_batch_size 32 --num_train_epoch 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64  --use_cls_vector --overwrite_cache  > experiments/logs/rebert_1.log &

# on test
acc = 0.8363003095975232


#  use cls_vector, use entity_vector
CUDA_VISIBLE_DEVICES="1" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_2 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_cls_vector --use_entity_vector  --overwrite_cache  > experiments/logs/rebert_2.log &

# on test
acc = 0.8448142414860681


```


### ablation on the span identification method (default = v2 (add entity markers))

```bash

#  span_identification_method = v1
CUDA_VISIBLE_DEVICES="1" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_3 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --span_identification_method v1 --overwrite_cache > experiments/logs/rebert_3.log &

# on test
acc = 0.8375241779497099


#  span_identification_method = v1； 加上了relation的头尾实体embedding；
CUDA_VISIBLE_DEVICES="2" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_4 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --span_identification_method v1 --use_rel_position_embedding --overwrite_cache > experiments/logs/rebert_4.log &

# on test
acc = 0.8344294003868472

#  span_identification_method = v3
CUDA_VISIBLE_DEVICES="2" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_5 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --span_identification_method v3 --overwrite_cache > experiments/logs/rebert_5.log &

# on test
acc = 0.8375241779497099


```



### ablation on the mention pooling operation (default = start)
```bash

# use mention_pooling = avg
CUDA_VISIBLE_DEVICES="3" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_6 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --mention_pooling avg --overwrite_cache > experiments/logs/rebert_6.log &

# on test
re_acc = 0.8417182662538699

# use mention_pooling = max
CUDA_VISIBLE_DEVICES="3" nohup python -u main.py --data_dir ./data/ --task semeval10 --model_type bert --model_name_or_path bert-base-uncased --model_dir experiments/outputs/rebert_7 --do_train --do_eval --train_batch_size 32 --num_train_epochs 8 --learning_rate 5e-5 --linear_learning_rate 5e-4 --warmup_steps 200 --max_seq_len 64 --use_entity_vector --mention_pooling max --overwrite_cache > experiments/logs/rebert_7.log &

# on test
re_acc = 0.8401702786377709

```

