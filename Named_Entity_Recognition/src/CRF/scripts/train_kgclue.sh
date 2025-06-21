
task_name=kgclue
seed=112101
sub_token_label_scheme=v3

echo "${task_name}"
echo "${seed}"


CUDA_VISIBLE_DEVICES="0" python -u src/CRF/main.py --seed ${seed} --task_name $task_name --labels datasets/kgclue/ner_data/crf_bio_labels.txt --is_flat_ner --data_dir datasets/kgclue/ner_data --model_type bertcrf --model_name_or_path resources/chinese_bert_wwm_ext --output_dir experiments/CRF/outputs/bert_CRF_${task_name}_${seed} --max_seq_length 128 --do_train --do_eval --do_predict --evaluate_during_training --do_lower_case --per_gpu_train_batch_size 128  --per_gpu_eval_batch_size 64 --learning_rate 2e-5 --crf_learning_rate 1e-3 --num_train_epochs 20 --warmup_steps 150 --logging_steps 100 --save_steps 200 --eval_all_checkpoints --overwrite_output_dir --gradient_checkpointing --log_dir experiments/CRF/logs/bert_CRF_${task_name}_${seed}.log --ignore_index 0 --overwrite_cache --sub_token_label_scheme ${sub_token_label_scheme}

# > experiments/CRF/logs/bert_CRF_${task_name}_${seed}.log &
