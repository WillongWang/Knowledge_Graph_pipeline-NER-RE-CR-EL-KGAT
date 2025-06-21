
task_name=conll03
seed=102401

echo "${task_name}"
echo "${seed}"

CUDA_VISIBLE_DEVICES="0" python src/MLP/main.py --seed ${seed} --task_name ${task_name} --labels datasets/${task_name}/bio_labels.txt --is_flat_ner --data_dir datasets/${task_name}/ --model_type bertmlp --model_name_or_path resources/bert_base_uncased --output_dir experiments/MLP/outputs/bert_MLP_${task_name}_${seed} --max_seq_length 128 --do_train --do_eval --do_predict --evaluate_during_training --do_lower_case --per_gpu_train_batch_size 32  --per_gpu_eval_batch_size 64 --learning_rate 2e-5 --classifiers_learning_rate 1e-4 --num_train_epochs 20 --warmup_steps 200 --logging_steps 200 --save_steps 200 --eval_all_checkpoints --overwrite_output_dir --gradient_checkpointing --log_dir experiments/MLP/logs/bert_MLP_${task_name}_${seed}.log --ignore_index -100 --overwrite_cache
