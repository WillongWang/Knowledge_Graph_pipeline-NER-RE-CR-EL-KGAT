import argparse

import sys
sys.path.append("./")

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES
from data_loader import load_and_cache_examples


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("dev")
        trainer.evaluate("test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str, help="Path of pre-trained models; ")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--re_label_file", default="semeval_rel2id.json", type=str, help="RE Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float,
                        help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--overwrite_cache', action="store_true",
                        help="whether to ignore cached file!")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # parser.add_argument("--ignore_index", default=0, type=int,
    #                     help='Specifies a target value that is ignored and does not contribute to the input gradient')

    # linear 层
    parser.add_argument(
        "--linear_learning_rate", default=5e-5, type=float, help="The initial learning rate for CRF layer."
    )

    # RE option
    parser.add_argument(
        "--use_bert_pooler", action="store_true",
        help="Whether to use bert pooler for hidden representations of two entities"
    )
    parser.add_argument(
        "--use_cls_vector", action="store_true",
        help="Whether to use the whole sent's representation as input features"
    )
    parser.add_argument(
        "--use_entity_vector", action="store_true",
        help="Whether to use the entity's representation as input features"
    )

    parser.add_argument(
        "--use_focal_loss", action="store_true",
        help="Whether to use focal loss as the loss objective"
    )
    parser.add_argument(
        "--focal_loss_gamma", default=2.0, type=float,
        help="gamma in focal loss"
    )
    parser.add_argument(
        "--class_weights", default=None, type=float,
        help="class_weights, written in string like '1.0,1.0,1.0,2.0,1.0' "
    )

    # 针对RE任务 span identitfication的 选项
    parser.add_argument(
        "--span_identification_method", default="v2", type=str,
        help="v1: 不做改变；"
             "v2:  添加entity markers"
             "v3:  替换为实体符号",
    )
    parser.add_argument(
        "--use_rel_position_embedding", action="store_true",
        help="是否使用面向关系抽取的实体位置编码"
    )

    parser.add_argument("--mention_pooling", default="start", type=str,
                        help="mention pooling should be in type selected in the list: [start, avg, max]" )

    parser.add_argument("--include_nli_ops", action="store_true",
                        help="features = torch.cat([head_hidden, tail_hidden, head_hidden*tail_hidden, head_hidden-tail_hidden], 1)")

    args = parser.parse_args()

    main(args)
