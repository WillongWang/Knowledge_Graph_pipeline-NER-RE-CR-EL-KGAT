import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, compute_metrics, get_re_labels

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.re_label2id = get_re_labels(args)

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        finetuning_task=args.task,
                                                        gradient_checkpointing=True,
                                                        )
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      re_label2id=self.re_label2id)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        for n, p in self.model.named_parameters():
            print(n)

        # Prepare optimizer and schedule (linear warmup and decay)

        optimizer_grouped_parameters = []

        # BERT部分参数，设置一个较低的学习率
        bert_params = list(self.model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.learning_rate,
            }
        ]

        # 线性层参数
        linear_params = list(self.model.classifier.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in linear_params if not any(nd in n for nd in no_decay)],
                'weight_decay': self.args.weight_decay,
                "lr": self.args.linear_learning_rate,
            },
            {
                'params': [p for n, p in linear_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.args.linear_learning_rate,
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'rel_position_ids': batch[3],
                          're_label_ids': batch[4],
                          'head_entity_pos': batch[5],
                          'tail_entity_pos': batch[6],
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        re_preds = None
        out_re_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'rel_position_ids': batch[3],
                          're_label_ids': batch[4],
                          'head_entity_pos': batch[5],
                          'tail_entity_pos': batch[6],
                          }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, re_logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if re_preds is None:
                re_preds = re_logits.detach().cpu().numpy()
                out_re_label_ids = inputs['re_label_ids'].detach().cpu().numpy()
            else:
                re_preds = np.append(re_preds, re_logits.detach().cpu().numpy(), axis=0)
                out_re_label_ids = np.append(
                    out_re_label_ids, inputs['re_label_ids'].detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        re_preds = np.argmax(re_preds, axis=1)

        total_result = compute_metrics(re_preds, out_re_label_ids)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          re_label2id=self.re_label2id)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
