from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input:[N, C]
        :param target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic
    used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient
    between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)

    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is
            \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the
                input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: OHEM（online hard example miniing）
                max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: float = 0.0,
                 alpha: float = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position=True) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        # input: [N, C]
        flat_input = input
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        # [N, ] --> [N, C]
        flat_target = F.one_hot(target, num_classes=logits_size).float() \
            if self.index_label_position else target.float()

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0:
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                # logits_size: 类别数
                # pos_example： 标签为 label_idx的是正例；
                # neg_example： 标签不为 label_idx的是负例；
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    # masked_select： 返回一个1-D tensor，根据flat_input对应于mask=1的部分的值返回
                    # 得到负样本对于本标签的打分
                    neg_scores = torch.masked_select(
                        flat_input,
                        neg_example.view(-1, 1).bool()
                    ).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]

                    # 预测为本标签或者正确标签是本标签
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:, label_idx] >= threshold) | pos_example.view(-1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(
                    flat_input_idx.view(-1, 1),
                    flat_target_idx.view(-1, 1)
                )
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]  # 概率值
                flat_target_idx = flat_target[:, label_idx]  # 有正样本，有负样本 【N, 】

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)