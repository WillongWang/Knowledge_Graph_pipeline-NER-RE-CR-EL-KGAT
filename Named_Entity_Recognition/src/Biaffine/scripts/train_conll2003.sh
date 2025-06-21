
task_name=conll03
seed=102401

echo "${task_name}"
echo "${seed}"

CUDA_VISIBLE_DEVICES="1" nohup python src/Biaffine/main.py --seed ${seed} --task_name $task_name --is_flat_ner --data_dir datasets/${task_name}/ --model_type bert_biaffine --model_name_or_path resources/bert_base_uncased --output_dir experiments/biaffine/outputs/bert_biaffine_${task_name}_${seed} --labels datasets/${task_name}/biaffine_labels.txt --max_seq_length 128 --do_train --do_eval --do_predict --evaluate_during_training --do_lower_case --per_gpu_train_batch_size 32  --per_gpu_eval_batch_size 64 --learning_rate 2e-5 --num_train_epochs 20 --warmup_steps 200 --logging_steps 200 --save_steps 200 --eval_all_checkpoints --overwrite_output_dir --gradient_checkpointing --biaffine_learning_rate 2e-4 --log_dir experiments/biaffine/logs/bert_biaffine_${task_name}_${seed}.log > experiments/biaffine/logs/bert_biaffine_${task_name}_${seed}.nohup_log &
