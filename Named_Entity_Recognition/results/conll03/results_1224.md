## conll03 数据集结果


### 模型： BERT+MLP


#### 使用不同的 sub-token 赋予NER label的方案

方案 v1： 
dev f1 = 0.9432539002228699
test f1 = 0.9079843764192933

experiments/MLP/outputs/bert_MLP_conll03_101001/eval_results.txt
f1_experiments/MLP/outputs/bert_MLP_conll03_101001 = 0.9419929740382142
f1_experiments/MLP/outputs/bert_MLP_conll03_101001/checkpoint = 0.943011397720456

experiments/MLP/outputs/bert_MLP_conll03_101001/test_results.txt
0.9063208488256098

方案 v2： 
dev f1 = 0.8670176630434783
test f1 = 0.8596412151807445

方案 v3： 
dev f1 = 0.9455607075390693
test f1 = 0.9048483747957147