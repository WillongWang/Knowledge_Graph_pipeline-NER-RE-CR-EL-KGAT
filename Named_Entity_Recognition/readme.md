## CPEE-BERT: confidence and patience based BERT early exiting for NER

# A-word : B-PER;
    # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, O, O; (sub_token_label_scheme == "v2")
    # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, PAD, PAD; (sub_token_label_scheme == "v3",loss ignore [PAD])
    # A-word --> BERT tokenize --> A_1, A_2, A_3: B-PER, I-PER, I-PER; (sub_token_label_scheme == "v1")

early exit for NER

nest entity: Bert + Biaffine + Pabee (已改完)

flat entity: Bert + CRF + Pabee (完成中)

### dataset list

| 名称          | 语种    | 实体类别(不加"O") | 是否含有嵌套实体/占比 | 样本数 |
| ------------- | ------- | ----------------- | --------------------- | ------ |
| ACE 2004      | English | 7                 | 是                    | 5586   |
| ACE 2005      | English | 7                 | 是                    | 6484   |
| GENIA         | English | 5                 | 是/17%                | 14835  |
| CoNLL 2003    | English | 4                 | 否                    | 10779  |
| OntoNotes 4.0 | Chinese | 4                 | 否                    | 5977   |
| MSRA          | Chinese | 3                 | 否                    | 26087  |

链接：https://pan.baidu.com/s/1FZTxRbeS2PgigSn7O7fvpA 
提取码：emub 

### pretrained model

bert-base-chinese

bert-base-uncased

### input format

MRC data format

reference: https://github.com/ShannonAI/mrc-for-flat-nested-ner

### run the code

main.py 

### file structure

```
 ├── data
 |  └── ace2004
 |  └── ace2005
 |  └── conll03
 |  └── genia
 |  └── zh_msra
 |  └── zh_onto4(数据名就是任务名称) 
 ├── log
 ├── losses
 ├── model
 ├── output(训练完的模型参数)
 ├── pretrained
 |  └── bert-base-chinese
 |  └── bert-base-uncased
 ├── processorss
 |  └── ner_seq.py(数据预处理)
 |  └── martix_seq_judgement.py(早退条件)
 |  └── utils_ner.py
 ├── result(结果文档)
 ├── runs(tensorboard)
 ├── scrips(bash脚本)
 ├── src(transformers)
 ├── tools
 |  └── common.py
 |  └── finetuning_argparse.py
 ├── main.py
 ├── readme.md
```

###  results

result/result_PabeeBiaffine_20210912