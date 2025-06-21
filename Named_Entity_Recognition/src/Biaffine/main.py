# coding=utf-8

from __future__ import absolute_import, division, print_function
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import sys
import json
import glob
import logging
import os
import warnings
from tqdm import tqdm, trange

warnings.filterwarnings("ignore")
sys.path.append("./")

from src.CRF.ner_data_utils import read_example_from_CCKS, read_example_form_MRC, get_labels

from src.transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from src.transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
from src.transformers import RobertaConfig
from src.transformers import DistilBertConfig

from src.Biaffine.finetuning_argparse import get_argparse

from src.Biaffine.generic_utils import set_seed
from src.Biaffine.ner_data_utils import ner_processors as processors
from src.Biaffine.ner_data_utils import convert_MRC_examples_to_features
from src.Biaffine.biaffine_utils import transform_label_matrix2spans, \
    Rm2entities, count_same_entities

from src.Biaffine.modeling_bert import BertForTokenClassificationBiaffine

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import numpy as np

logger = logging.getLogger()

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, DistilBertConfig)),
    ())
MODEL_CLASSES = {
    "bert_biaffine": (BertConfig, BertForTokenClassificationBiaffine, BertTokenizer),
}  # important


def train(args, train_dataset, model, tokenizer, labels):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 \
        else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    for n, p in model.named_parameters():
        print("model's parameters:", n)
    optimizer_grouped_parameters = []
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    optimizer_grouped_parameters += [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]

    if args.use_lstm:  # 是否需要10*
        linear1_param_optimizer = list(model.lstm.named_parameters())
        optimizer_grouped_parameters += [
            {'params': [p for n, p in linear1_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay,
             'lr': args.lstm_learning_rate},
            {'params': [p for n, p in linear1_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': args.lstm_learning_rate}
        ]

    biaffine_param_optimizer = list(model.biaffine_classifiers.named_parameters())
    optimizer_grouped_parameters += [
        {'params': [p for n, p in biaffine_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.biaffine_learning_rate
         },
        {'params': [p for n, p in biaffine_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.biaffine_learning_rate}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_score = 0
    temp_score = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
            outputs = model(**inputs)
            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    # log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:
                        _, results = evaluate(args, model, tokenizer, labels, mode="dev")
                        temp_score = results["f1"]

                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                            print("eval_{}".format(key), value, global_step)
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps

                    learning_rate_scalar = scheduler.get_lr()[0]
                    # learning_rate_biaffine = scheduler.get_lr()[2]
                    logs["learning_rate"] = learning_rate_scalar
                    # logs["learning_rate_biaffine"] = learning_rate_biaffine
                    logs["loss"] = loss_scalar
                    print("lr", learning_rate_scalar, global_step)
                    print("loss", loss_scalar, global_step)
                    logging_loss = tr_loss
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    ###保存最优解
                    if temp_score > best_score:
                        best_score = temp_score
                        outputs_dir = os.path.join(args.output_dir,
                                                   "checkpoint")  # join(args.output_dir, "checkpoint-{}".format(global_step))
                        if not os.path.exists(outputs_dir):
                            os.makedirs(outputs_dir)
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(outputs_dir)
                        tokenizer.save_pretrained(outputs_dir)
                        torch.save(args, os.path.join(outputs_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", outputs_dir)
                        torch.save(optimizer.state_dict(), os.path.join(outputs_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(outputs_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", outputs_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels_list, mode, prefix="", file_to_predict=None, patience=0):
    if args.model_type == "albert":
        model.albert.set_regression_threshold(args.regression_threshold)
        model.albert.set_patience(patience)
        model.albert.reset_stats()
    elif args.model_type == "bert_biaffine":
        model.bert.set_regression_threshold(args.regression_threshold)
        model.bert.set_patience(patience)
        model.bert.reset_stats()
    else:
        raise NotImplementedError()
    eval_dataset = load_and_cache_examples(args, tokenizer, labels_list, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # NOte that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    total_nb_correct = 0
    total_nb_pred = 0
    total_nb_true = 0
    list_predicted_samples = []

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]
                      }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs
            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        # add
        # get predicted spans in a batch
        input_ids = inputs['input_ids']
        labels = inputs['labels'].to_dense()
        preds = logits
        masks = inputs['attention_mask']
        for i in range(labels.shape[0]):
            input_id = input_ids[i]
            label = labels[i]
            pred = preds[i]
            mask = masks[i]
            # cutting the padding
            true_len = int(mask.sum().item())
            input_id = input_id[:true_len]
            pred = pred[:true_len, :true_len]
            label = label[:true_len, :true_len]
            labeled_items = transform_label_matrix2spans(label, id2label_map=labels_list)
            labeled_items = sorted(labeled_items, key=lambda x: x[0], reverse=False)
            # print("labeled_items: ", labeled_items)
            pred_items = Rm2entities(
                pred, is_flat_ner=args.is_flat_ner, id2label_map=labels_list
            )  # is_flat_ner=True
            pred_items = sorted(
                pred_items, key=lambda x: x[0], reverse=False
            )
            # print('pred_items:', pred_items)
            same_num = count_same_entities(labeled_items, pred_items)
            pred_num = len(pred_items)
            label_num = len(labeled_items)
            # 将预测结果记录下来
            list_predicted_samples.append(
                {
                    "ground_truth": list(labeled_items),
                    "prediction": list(pred_items),
                    "input_sequence": input_id.detach().cpu().numpy().tolist(),
                }
            )
            total_nb_correct += same_num
            total_nb_pred += pred_num
            total_nb_true += label_num
    eval_loss = eval_loss / nb_eval_steps
    precision = total_nb_correct / total_nb_pred if total_nb_pred > 0 else 0
    recall = total_nb_correct / total_nb_true if total_nb_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    results = {
        "loss": eval_loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info(" %s = %s", key, str(results[key]))
        print(" %s = %s" %( key, str(results[key])))

    if file_to_predict:  # 预测结果的记录，是不是可以解码成英文字符之后再输出？
        json.dump(
            list_predicted_samples,
            open(file_to_predict, 'w', encoding="utf-8"),
            ensure_ascii=False,
        )
    if args.eval_all_checkpoints and patience != 0:
        if args.model_type == "albert":
            model.albert.log_stats()
        elif args.model_type == "bert_biaffine":
            model.bert.log_stats()
        else:
            raise NotImplementedError()
    return eval_loss, results


def load_and_cache_examples(args, tokenizer, labels, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    cached_features_file = os.path.join(args.data_dir, "cache_{}_{}_{}_{}".format(
        args.model_type,
        mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = get_labels(args.labels)
        if args.task_name.lower() in ["ccks", "kgclue", ]:
            examples = read_example_from_CCKS(args.data_dir, mode)
        else:
            examples = read_example_form_MRC(args.data_dir, mode)

        features = convert_MRC_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                    cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args.model_type in [
                                                        "xlnet"] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args.model_type in ["roberta"]),
                                                    pad_on_left=bool(args.model_type in ["xlnet"]),
                                                    pad_token=
                                                    tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args.model_type in [
                                                        "xlnet"] else 0,
                                                    pad_token_label_id=args.pad_token_label_id
                                                    )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids_biaffine = torch.stack([f.label_ids_biaffine for f in features])
    # print('all_label_ids_biaffine:',all_label_ids_biaffine[0])
    dataset = TensorDataset(  # zip:
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids_biaffine)
    return dataset


def main():
    args = get_argparse().parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://
        import ptvsd
        print("waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirext_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA， GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info({"n_gpu: ": args.n_gpu})
    else:  # initializes the distributed backend which will take care of suchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_porcess_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=args.log_dir)

    logging.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    # set seed
    set_seed(args)
    # prepare NER task

    labels_list = get_labels(args.labels)
    print("labels: ", labels_list)
    label_map = {label: i for i, label in enumerate(labels_list)}
    num_labels = len(labels_list)

    # Use cross entropy ignore index as padding label id so that only real ids contribute to the los later
    # pad_token_label_id = CrossEntropyLoss().ignore_index
    pad_token_label_id = label_map["O"]
    args.pad_token_label_id = pad_token_label_id
    if args.patience not in ["0", "-1"] and args.per_gpu_eval_batch_size != 1:
        raise ValueError("The eval batch size must be 1 with PABEE inference on.")

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None,
                                          output_dropout=args.output_dropout,

                                          )
    config.simplify_biaffine = args.simplify_biaffine

    config.sampling_ratio = args.sampling_ratio
    config.use_focal = args.use_focal
    config.use_ffn = args.use_ffn
    config.use_lstm = args.use_lstm
    config.gradient_checkpointing = args.gradient_checkpointing
    config.output_dropout = args.output_dropout

    config.do_hard_negative_sampling = args.do_hard_negative_sampling
    config.hns_multiplier = args.hns_multiplier


    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        labels_list=labels_list, )
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model& vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels_list, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels_list, )
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s.", args.output_dir)
        # Saving a trained model, configuration and tokenizer using save_pretrained()
        # They can then be reload using from_pretrained()
        model_to_save = model.module if hasattr(model,
                                                "module") else model  # Take care of disrtibuted/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, labels_list=labels_list, )
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    best_checkpoint = None
    best_score = 0.0
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        patience_list = [int(x) for x in args.patience.split(",")]
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        print(checkpoints)
        if args.eval_all_checkpoints:
            checkpoints += list(
                os.path.dirname(c) for c in
                sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if "\\" in checkpoint:
                prefix = checkpoint.split("\\")[-1] if checkpoint.find("checkpoint") != -1 else ""
            else:
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint, labels_list=labels_list,  )
            model.to(args.device)
            print(f"Evaluation for checkpoint {prefix}")
            for patience in patience_list:
                loss, result = evaluate(args, model, tokenizer, labels_list, mode="dev", prefix=prefix,
                                        patience=patience)
                if result['f1'] > best_score:
                    best_score = result['f1']
                    best_checkpoint = checkpoint  # 记录f1分数最高的checkpoint,不管是第几个patience
                if global_step:
                    result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
                results.update(result)  # results记录的是最后一个checkpoint的最后一个patience的数值
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best_score = {}\n".format(best_score))
            writer.write("best_checkpoint = {}\n".format(str(best_checkpoint)))  # 有可能会报错

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        if not best_checkpoint:
            best_checkpoint = args.output_dir
        tokenizer = tokenizer_class.from_pretrained(args.output_dir,
                                                    do_lower_case=args.do_lower_case)  # output_dir应该是最好的模型的路径
        model = model_class.from_pretrained(best_checkpoint, labels_list=labels_list, )
        model.to(args.device)
        output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.json")
        result, predictions = evaluate(
            args, model,
            tokenizer,
            labels_list,
            mode="test",
            file_to_predict=output_test_predictions_file,
            patience=int(args.patience)
        )
        # save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, 'w') as writer:
            for key in sorted(predictions.keys()):
                writer.write("{} = {}\n".format(key, str(predictions[key])))
    return results


if __name__ == '__main__':
    main()
