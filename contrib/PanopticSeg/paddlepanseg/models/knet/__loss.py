# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def reduce_loss(loss: paddle.Tensor, reduction: str) -> paddle.Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    # reduction_enum = F._Reduction.get_enum(reduction)
    reduction_enum = reduction
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == "none":
        return loss
    elif reduction_enum == "mean":
        return loss.mean()
    elif reduction_enum == "sum":
        return loss.sum()


def weight_reduce_loss(loss: paddle.Tensor,
                       weight: Optional[paddle.Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> paddle.Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            # eps = paddle.finfo(paddle.float32).eps
            eps = 1.1920928955078125e-07
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (paddle.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (paddle.Tensor): The learning label of the prediction.
        weight (paddle.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        paddle.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses
    loss = F.cross_entropy(
        pred.transpose([0, 2, 3, 1]),   # TODO: This is not the same as the mmdet
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.cast("float32")
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).cast("float32")
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (paddle.Tensor): The prediction with shape (N, 1) or (N, ).
            When the shape of pred is (N, 1), label will be expanded to
            one-hot format, and when the shape of pred is (N, ), label
            will not be expanded to one-hot format.
        label (paddle.Tensor): The learning label of the prediction,
            with shape (N, ).
        weight (paddle.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        paddle.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    if pred.dim() != label.dim():
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.size(-1), ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).cast("float32")
        if weight is not None:
            # The inplace writing method will have a mismatched broadcast
            # shape error if the weight and valid_mask dimensions
            # are inconsistent such as (B,N,1) and (B,N,C).
            weight = weight * valid_mask
        else:
            weight = valid_mask

    # average loss over non-ignored elements
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = valid_mask.sum().item()

    # weighted element-wise losses
    weight = weight.cast("float32")
    loss = F.binary_cross_entropy_with_logits(
        pred, label.cast("float32"), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (paddle.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (paddle.Tensor): The learning label of the prediction.
        label (paddle.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        paddle.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = paddle.arange(0, num_rois, dtype="int64")
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]



class CrossEntropyLoss(nn.Layer):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (paddle.Tensor): The prediction.
            label (paddle.Tensor): The learning label of the prediction.
            weight (paddle.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            paddle.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label.cast("int64"),
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls
    
    
    
    
    
    
    
    
    
    
    
# ------------------------------------------- Focal loss -------------------------------------------
    
    
    
# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = F.sigmoid(pred)
    target = target.cast(pred.dtype)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.shape[0] == loss.shape[0]:
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.reshape([-1, 1])
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
            The target shape support (N,C) or (N,), (N,C) means one-hot form.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       avg_factor=None):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class FocalLoss(nn.Layer):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if pred.dim() == target.dim():
                    # this means that target is already in One-Hot form.
                    calculate_loss_func = py_sigmoid_focal_loss
                elif False:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.shape[1]
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls
    
    


# ---------------------------------------------------- dice_loss ----------------------------------------------------



def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).cast("float32")

    a = paddle.sum(input * target, 1)
    if naive_dice:
        b = paddle.sum(input, 1)
        c = paddle.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = paddle.sum(input * input, 1) + eps
        c = paddle.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class DiceLoss(nn.Layer):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = F.sigmoid(pred)
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss
    
    
LOSS = {
    'CrossEntropyLoss' : CrossEntropyLoss,
    'FocalLoss'        : FocalLoss,
    'DiceLoss'         : DiceLoss,
}


def build_loss(cfg, *args, **kwargs):
    """Build padding layer.

    Args:
        cfg (None or dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    loss_type = cfg_.pop('type')
    if loss_type not in LOSS:
        raise KeyError(f'Unrecognized padding type {loss_type}.')
    else:
        loss_layer = LOSS.get(loss_type)

    layer = loss_layer(*args, **kwargs, **cfg_)

    return layer



def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.shape[0] == 0:
        # accu = [pred.new_tensor(0.) for i in range(len(topk))]
        accu = [paddle.zeros([0]).cast(pred.dtype) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.shape[0] == target.shape[0]
    assert maxk <= pred.shape[1], \
        f'maxk {maxk} exceeds pred dimension {pred.shape[1]}'
    pred_value, pred_label = pred.topk(maxk, axis=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.equal(target.reshape([1, -1]).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].cast("float32").flatten().sum(0, keepdim=True)
        res.append(correct_k * (100.0 / pred.shape[0]))
    return res[0] if return_single else res
