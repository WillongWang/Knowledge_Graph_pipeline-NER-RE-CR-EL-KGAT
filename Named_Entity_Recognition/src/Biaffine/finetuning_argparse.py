# coding=utf-8

import argparse

def get_argparse():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="name of the task.")
    parser.add_argument("--is_flat_ner", action="store_true",
                        help="whether the task is flat, or not (nested).")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir should contain the training files for the CoNLL-2003 NER task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output dir where the model predictions and checkpoints will be written.")

    ## other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name.")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same model_name.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do yu want to store the pre-trained models downloaded from s3.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization.Sequence longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    ## model configurations
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward.update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate_layer_decay", default=0.9, type=float,
                        help="ratio of layerwise lr decay, i.e., lower layer gets smaller lr.")

    # output_dropout
    parser.add_argument("--output_dropout", default=0.1, type=float,
                        help="Dropout rate outside the bert encoder.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epoches to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", default=50, type=int,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", default=50, type=int,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CuDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory.")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets.")
    parser.add_argument("--seed", default=42, type=int,
                        help="random seed for initialization")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Whether to use gradient_checkpointing.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit mixed precision through NVIDIA apex instead of 32-bit.")
    parser.add_argument("--fp16_opt_level", default="01", type=str,
                        help="For fp: Apex AMP optimization level selected in ['00', '01', '02', and '03'].")
    parser.add_argument("--local_rank", default=-1, type=int,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", default="", type=str, help="For distant debugging.")
    parser.add_argument("--server_port", default="", type=str, help="For distant debugging.")

    # lstm
    parser.add_argument("--use_lstm", action="store_true",
                        help="Whether to use lstm to extract the features")
    parser.add_argument("--lstm_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for LSTM layer.")

    # ffn
    parser.add_argument("--use_ffn", action="store_true",
                        help="whether to use ffn before the biaffine layer")
    parser.add_argument("--ffn_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for ffn.")

    # biaffine
    parser.add_argument("--biaffine_learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for biaffine.")
    parser.add_argument("--use_focal", action="store_true",
                        help="Whether to use focal loss")

    # pabee
    parser.add_argument("--patience", default="0", type=str, required=False, )
    parser.add_argument("--regression_threshold", default=0, type=float, required=False, )
    parser.add_argument("--log_dir", default='log/', type=str, required=True,
                        help="The log directions.")

    parser.add_argument('--simplify_biaffine', action='store_true', help='Whether to simplify the biaffine model')

    parser.add_argument('--sampling_ratio', default=0.1, type=float,
                        help="Randomly sample a portion of logits")

    parser.add_argument('--do_hard_negative_sampling', action='store_true', help='do_hard_negative_sampling', )

    parser.add_argument('--hns_multiplier', default=None, type=float, help='in do_hard_negative_sampling', )

    return parser