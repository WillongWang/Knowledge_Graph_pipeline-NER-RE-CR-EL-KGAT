
task_name=ace2004
seed=122401
sampling_ratio=0.1

echo "${task_name}"
echo "${seed}"
echo "${sampling_ratio}"

CUDA_VISIBLE_DEVICES="3" nohup python src/Biaffine/main.py --seed ${seed} --task_name $task_name --data_dir datasets/${task_name}/ --model_type bert_biaffine --model_name_or_path resources/bert_base_uncased --output_dir experiments/biaffine/outputs/bert_biaffine_${task_name}_${seed} --labels datasets/${task_name}/biaffine_labels.txt --max_seq_length 128 --do_train --evaluate_during_training --do_lower_case --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 --learning_rate 2e-5 --num_train_epochs 50 --warmup_steps 200 --logging_steps 200 --save_steps 200 --eval_all_checkpoints --overwrite_output_dir --gradient_checkpointing --biaffine_learning_rate 2e-4 --log_dir experiments/biaffine/logs/bert_biaffine_${task_name}_${seed}.log > experiments/biaffine/logs/bert_biaffine_${task_name}_${seed}.nohup_log &

# python src/Biaffine/main.py --seed 110601 --task_name ace2004 --is_flat_ner --data_dir datasets/ace2004/ --model_type bert_biaffine --model_name_or_path resources/bert_base_uncased --output_dir experiments/biaffine/outputs/bert_biaffine_ace2004_110601 --labels datasets/ace2004/biaffine_labels.txt --max_seq_length 128 --do_train --evaluate_during_training --do_lower_case --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 64 --learning_rate 2e-5 --num_train_epochs 15 --warmup_steps 100 --logging_steps 100 --save_steps 200 --eval_all_checkpoints --overwrite_output_dir --gradient_checkpointing --biaffine_learning_rate 2e-4 --log_dir experiments/biaffine/logs/bert_biaffine_ace2004_110601.log
