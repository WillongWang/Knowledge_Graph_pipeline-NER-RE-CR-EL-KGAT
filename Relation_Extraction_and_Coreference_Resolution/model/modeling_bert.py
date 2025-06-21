import copy

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF

from .focal_loss import FocalLoss
from .module import Classifier


class ReBERT(BertPreTrainedModel):
    def __init__(self, config, args, re_label2id):
        super(ReBERT, self).__init__(config)
        self.args = args
        self.num_re_labels = len(re_label2id)
        self.bert = BertModel(config=config)  # Load pretrained bert

        if self.args.use_cls_vector and self.args.use_entity_vector:
            input_size_for_classifier = config.hidden_size * 3
        elif not self.args.use_cls_vector and self.args.use_entity_vector:
            if self.args.include_nli_ops:
             input_size_for_classifier = config.hidden_size * 4
            else:
             input_size_for_classifier = config.hidden_size * 2
        else:
            input_size_for_classifier = config.hidden_size * 1

        # 面向关系的位置编码
        self.rel_position_embedding = None
        if self.args.use_rel_position_embedding:
            self.rel_position_embedding = nn.Embedding(
                3, config.hidden_size,
            )

        # mention_pooling: 实体向量特征的计算
        self.mention_pooling = self.args.mention_pooling

        self.classifier = Classifier(
            input_size_for_classifier,
            self.num_re_labels,
            args.dropout_rate
        )

        # class weights
        self.alpha = None
        if self.args.class_weights:
            self.alpha = self.args.class_weights.split(",")
            self.alpha = [w.strip() for w in self.alpha]
            self.alpha = [w for w in self.alpha if w]
            self.alpha = [float(w) for w in self.alpha]
            self.alpha = torch.FloatTensor(self.alpha)

    def forward(self, input_ids,
                attention_mask,
                token_type_ids,
                rel_position_ids,
                re_label_ids,
                head_entity_pos=None,
                tail_entity_pos=None,
                ):

        input_embeddings = self.bert.embeddings.word_embeddings(
            input_ids,
            # token_type_ids=token_type_ids,
        )

        if self.rel_position_embedding is not None:
            input_embeddings = input_embeddings + self.rel_position_embedding(rel_position_ids)

        # print("input_embeddings: ", input_embeddings.shape)
        # print("attention_mask: ", attention_mask.shape)
        outputs = self.bert(
            # input_ids=input_ids,
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # sequence_output, pooled_output, (hidden_states), (attentions)


        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]: (B, H)

        if self.mention_pooling == "start":
            onehot = torch.zeros(sequence_output.size()[: 2]).float()  # (B, L)
            if torch.cuda.is_available():
                onehot = onehot.cuda()

            onehot_head = onehot.scatter(1, head_entity_pos[:, : 1], 1)  # (B, L)
            onehot_tail = onehot.scatter(1, tail_entity_pos[:, : 1], 1)  # (B, L)
            head_hidden = (onehot_head.unsqueeze(2) * sequence_output).sum(1)  # (B, H)
            tail_hidden = (onehot_tail.unsqueeze(2) * sequence_output).sum(1)  # (B, H)

        elif self.mention_pooling == "avg":
            onehot_head = torch.zeros(sequence_output.size()[: 2]).float()  # (B, L)
            onehot_tail = torch.zeros(sequence_output.size()[: 2]).float()  # (B, L)
            if torch.cuda.is_available():
                onehot_head = onehot_head.cuda()
                onehot_tail = onehot_tail.cuda()

            for i in range(sequence_output.size()[0]):
                onehot_head[i][head_entity_pos[i][0]: head_entity_pos[i][1]] = 1
                onehot_tail[i][tail_entity_pos[i][0]: tail_entity_pos[i][1]] = 1

            # print("onehot_head: ", onehot_head)
            # print("onehot_tail: ", onehot_tail)

            head_length = onehot_head.sum(1)  # (B, )
            tail_length = onehot_tail.sum(1)  # (B, )

            # print("head_length: ", head_length)
            # print("tail_length: ", tail_length)

            # print((onehot_head.unsqueeze(2) * sequence_output)[: 4, : , : 16])

            head_hidden = (onehot_head.unsqueeze(2) * sequence_output).sum(1)  # (B, H)
            tail_hidden = (onehot_tail.unsqueeze(2) * sequence_output).sum(1)  # (B, H)

            head_hidden = torch.div(head_hidden, head_length.unsqueeze(1))
            tail_hidden = torch.div(tail_hidden, tail_length.unsqueeze(1))

            # print("head_hidden: ", head_hidden)
            # print("tail_hidden: ", tail_hidden)

        elif self.mention_pooling == "max":

            sequence_output_1 = torch.clone(sequence_output)
            for i in range(sequence_output_1.size()[0]):
                sequence_output_1[i][: head_entity_pos[i][0]] = -1e15
                sequence_output_1[i][head_entity_pos[i][1]: ] = -1e15

            sequence_output_2 = torch.clone(sequence_output)
            for i in range(sequence_output_2.size()[0]):
                sequence_output_2[i][: tail_entity_pos[i][0]] = -1e15
                sequence_output_2[i][tail_entity_pos[i][1]:] = -1e15

            # print("sequence_output_1: ", sequence_output_1)
            # print("sequence_output_2: ", sequence_output_2)

            head_hidden, _ = torch.max(sequence_output_1, 1)
            tail_hidden, _ = torch.max(sequence_output_2, 1)

            # print("head_hidden: ", head_hidden)
            # print("tail_hidden: ", tail_hidden)

        else:
            raise ValueError("unsupported mention_pooling type: {} !!!".format(self.mention_pooling))


        # 确定输入分类器的features
        if self.args.use_cls_vector and self.args.use_entity_vector:
            features = torch.cat([pooled_output, head_hidden, tail_hidden], 1)  # (B, 3*H)
        elif not self.args.use_cls_vector and self.args.use_entity_vector:
            if self.args.include_nli_ops:
                features = torch.cat([head_hidden, tail_hidden, head_hidden * tail_hidden, head_hidden - tail_hidden],
                                     1)
            else:
                features = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2*H)
        elif self.args.use_cls_vector and not self.args.use_entity_vector:
            features = pooled_output  # (B, 1*H)
        else:
            raise ValueError("use_cls_vector and use_entity_vector can not be False at the same time!!!")

        logits = self.classifier(features)
        outputs = (logits,) + outputs[2: ]

        # 1. re_label Softmax
        if re_label_ids is not None:
            if self.args.use_focal_loss:
                re_loss_fct = FocalLoss(
                    self.num_re_labels,
                    alpha=self.alpha,
                    gamma=self.args.focal_loss_gamma,
                    size_average=True
                )
            else:
                re_loss_fct = nn.CrossEntropyLoss()

            re_loss = re_loss_fct(
                logits.view(-1, self.num_re_labels),
                re_label_ids.view(-1)
            )

            outputs = (re_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
