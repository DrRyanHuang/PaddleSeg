
from abc import ABCMeta, abstractmethod
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import inspect

from assign_result import AssignResult

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None



class DiceCost(object):
    """DiceCost.

     Args:
         weight (int | float, optional): loss_weight
         pred_act (bool): Whether to activate the prediction
            before calculating cost

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self,
                 weight=1.,
                 pred_act=False,
                 act_mode='sigmoid',
                 eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode
        self.eps = eps

    def dice_loss(cls, input, target, eps=1e-3):
        input = input.reshape([input.shape[0], -1])
        target = target.reshape([target.shape[0], -1]).cast("float32")
        # einsum saves 10x memory
        # a = torch.sum(input[:, None] * target[None, ...], -1)
        a = paddle.einsum('nh,mh->nm', input, target)
        b = paddle.sum(input * input, 1) + eps
        c = paddle.sum(target * target, 1) + eps
        d = (2 * a) / (b[:, None] + c[None, ...])
        # 1 is a constance that will not affect the matching, so ommitted
        return -d

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            mask_preds = F.sigmoid(mask_preds)
        elif self.pred_act:
            mask_preds = mask_preds.softmax(dim=0)
        dice_cost = self.dice_loss(mask_preds, gt_masks, self.eps)
        return dice_cost * self.weight



class MaskCost(object):
    """MaskCost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., pred_act=False, act_mode='sigmoid'):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode

    def __call__(self, cls_pred, target):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            cls_pred = F.sigmoid(cls_pred)
        elif self.pred_act:
            cls_pred = F.softmax(cls_pred, axis=0)

        _, H, W = target.shape
        # flatten_cls_pred = cls_pred.view(num_proposals, -1)
        # eingum is ~10 times faster than matmul
        pos_cost = paddle.einsum('nhw,mhw->nm', cls_pred, target)
        neg_cost = paddle.einsum('nhw,mhw->nm', 1 - cls_pred, 1 - target)
        cls_cost = -(pos_cost + neg_cost) / (H * W)
        return cls_cost * self.weight



class FocalLossCost:
    """FocalLossCost.

     Args:
         weight (int | float, optional): loss_weight
         alpha (int | float, optional): focal_loss alpha
         gamma (int | float, optional): focal_loss gamma
         eps (float, optional): default 1e-12
         binary_input (bool, optional): Whether the input is binary,
            default False.

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import FocalLossCost
         >>> import torch
         >>> self = FocalLossCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3236, -0.3364, -0.2699],
                [-0.3439, -0.3209, -0.4807],
                [-0.4099, -0.3795, -0.2929],
                [-0.1950, -0.1207, -0.2626]])
    """

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = F.sigmoid(cls_pred)
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        # cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        cls_cost = paddle.index_select(pos_cost, gt_labels, axis=1) - paddle.index_select(neg_cost, gt_labels, axis=1)
        return cls_cost * self.weight

    def _mask_focal_loss_cost(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits
                in shape (num_query, d1, ..., dn), dtype=torch.float32.
            gt_labels (Tensor): Ground truth in shape (num_gt, d1, ..., dn),
                dtype=torch.long. Labels should be binary.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).cast("float32")
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
            1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
            1 - cls_pred).pow(self.gamma)

        cls_cost = paddle.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            paddle.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classfication logits.
            gt_labels (Tensor)): Labels.

        Returns:
            Tensor: Focal cost matrix with weight in shape\
                (num_query, num_gt).
        """
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)




MATCH_COST = {
    'FocalLossCost': FocalLossCost,
    'DiceCost': DiceCost,
    'MaskCost': MaskCost,
}


def build_match_cost(cfg, registry=MATCH_COST, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    # if not isinstance(registry, Registry):
    #     raise TypeError('registry must be an mmcv.Registry object, '
    #                     f'but got {type(registry)}')
    # if not (isinstance(default_args, dict) or default_args is None):
    #     raise TypeError('default_args must be a dict or None, '
    #                     f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')
    



# -------------------------- assigner --------------------------


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""




class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 mask_cost=dict(type='SigmoidCost', weight=1.0),
                 dice_cost=dict(),
                 boundary_cost=None,
                 topk=1):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        if boundary_cost is not None:
            self.boundary_cost = build_match_cost(boundary_cost)
        else:
            self.boundary_cost = None
        self.topk = topk

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.shape[0], bbox_pred.shape[0]

        # 1. assign -1 by default
        # assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
        #                                       -1,
        #                                       dtype=paddle.int64)
        # assigned_labels = bbox_pred.new_full((num_bboxes, ),
        #                                      -1,
        #                                      dtype=paddle.int64)
        assigned_gt_inds = paddle.full((num_bboxes,), -1, dtype=paddle.int64)
        assigned_labels = paddle.full((num_bboxes,), -1, dtype=paddle.int64)
        
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        if self.mask_cost.weight != 0:
            reg_cost = self.mask_cost(bbox_pred, gt_bboxes)
        else:
            reg_cost = 0
        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(bbox_pred, gt_bboxes)
        else:
            dice_cost = 0
        if self.boundary_cost is not None and self.boundary_cost.weight != 0:
            b_cost = self.boundary_cost(bbox_pred, gt_bboxes)
        else:
            b_cost = 0
        cost = cls_cost + reg_cost + dice_cost + b_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if self.topk == 1:
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        else:
            topk_matched_row_inds = []
            topk_matched_col_inds = []
            for i in range(self.topk):
                matched_row_inds, matched_col_inds = linear_sum_assignment(
                    cost)
                topk_matched_row_inds.append(matched_row_inds)
                topk_matched_col_inds.append(matched_col_inds)
                cost[matched_row_inds] = 1e10
            matched_row_inds = np.concatenate(topk_matched_row_inds)
            matched_col_inds = np.concatenate(topk_matched_col_inds)

        matched_row_inds = paddle.to_tensor(matched_row_inds)
        matched_col_inds = paddle.to_tensor(matched_col_inds)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)


BBOX_ASSIGNERS = {
    "MaskHungarianAssigner": MaskHungarianAssigner
}


def build_assigner(cfg, registry=BBOX_ASSIGNERS, default_args=None):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        if default_args is None or 'type' not in default_args:
            raise KeyError(
                '`cfg` or `default_args` must contain the key "type", '
                f'but got {cfg}\n{default_args}')
    # if not isinstance(registry, Registry):
    #     raise TypeError('registry must be an mmcv.Registry object, '
    #                     f'but got {type(registry)}')
    # if not (isinstance(default_args, dict) or default_args is None):
    #     raise TypeError('default_args must be a dict or None, '
    #                     f'but got {type(default_args)}')

    args = cfg.copy()

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**args)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')