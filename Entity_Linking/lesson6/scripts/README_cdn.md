# DynaBERT

数据集： LCQMC


## use chinese-bert-wwm-ext as backbone model


**Step 1.** 
Fine-tune the pretrained [BERT base model](https://huggingface.co/bert-base-uncased).

```
CUDA_VISIBLE_DEVICES="0" python DynaBERT/run_glue.py \
    --seed 11901 \
	--model_type bert \
	--task_name cdn \
	--do_train \
	--data_dir datasets/CBLUE_datasets/CHIP-CDN/training_data_0120 \
	--model_dir resources/chinese_bert_wwm_ext \
	--output_dir experiments/outputs/bert_cdn_11901_finetuning \
	--max_seq_length 48 \
	--learning_rate 2e-5 \
    --warmup_steps 300 \
	--per_gpu_train_batch_size 256 \
    --per_gpu_eval_batch_size 256 \
	--num_train_epochs 5 \
	--training_phase finetuning \
    --evaluate_during_training True \
    --logging_steps 200
```

## 在 CDN dev集上测试：召回+排序全流程

```
python eval_el.py --model_type bert --task_name cdn --data_dir datasets/CBLUE_datasets/CHIP-CDN/CHIP-CDN/CHIP-CDN_dev.json --max_seq_length 48 --per_gpu_eval_batch_size 128 --model_dir experiments/outputs/bert_cdn_11901_finetuning --output_dir experiments/outputs/bert_cdn_11901_finetuning/dev_predicted_results.txt
```

