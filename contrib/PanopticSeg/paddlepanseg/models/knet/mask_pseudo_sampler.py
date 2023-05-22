import paddle
import inspect

from abc import ABCMeta, abstractmethod
import numpy.random as rng

from util_mixins import NiceRepr
from assign_result import AssignResult
# from mmdet.core.bbox import demodata




class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers."""

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Sample negative samples."""
        pass

    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (Tensor): Boxes to be sampled from.
            gt_bboxes (Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        bboxes = bboxes[:, :4]

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = paddle.concat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = paddle.concat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result


class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(RandomSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                            add_gt_as_proposals)
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        """Randomly sample some negative samples."""
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)


class SamplingResult(NiceRepr):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return paddle.concat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    # @classmethod
    # def random(cls, rng=None, **kwargs):
    #     """
    #     Args:
    #         rng (None | int | numpy.random.RandomState): seed or state.
    #         kwargs (keyword arguments):
    #             - num_preds: number of predicted boxes
    #             - num_gts: number of true boxes
    #             - p_ignore (float): probability of a predicted box assigned to \
    #                 an ignored truth.
    #             - p_assigned (float): probability of a predicted box not being \
    #                 assigned.
    #             - p_use_label (float | bool): with labels or not.

    #     Returns:
    #         :obj:`SamplingResult`: Randomly generated sampling result.

    #     Example:
    #         >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
    #         >>> self = SamplingResult.random()
    #         >>> print(self.__dict__)
    #     """


    #     # make probabalistic?
    #     num = 32
    #     pos_fraction = 0.5
    #     neg_pos_ub = -1

    #     assign_result = AssignResult.random(rng=rng, **kwargs)

    #     # Note we could just compute an assignment
    #     bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
    #     gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

    #     if rng.rand() > 0.2:
    #         # sometimes algorithms squeeze their data, be robust to that
    #         gt_bboxes = gt_bboxes.squeeze()
    #         bboxes = bboxes.squeeze()

    #     if assign_result.labels is None:
    #         gt_labels = None
    #     else:
    #         gt_labels = None  # todo

    #     if gt_labels is None:
    #         add_gt_as_proposals = False
    #     else:
    #         add_gt_as_proposals = True  # make probabalistic?

    #     sampler = RandomSampler(
    #         num,
    #         pos_fraction,
    #         neg_pos_ub=neg_pos_ub,
    #         add_gt_as_proposals=add_gt_as_proposals,
    #         rng=rng)
    #     self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
    #     return self


class MaskSamplingResult(SamplingResult):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_masks': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_masks': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, masks, gt_masks, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_masks = masks[pos_inds]
        self.neg_masks = masks[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_masks.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if self.pos_assigned_gt_inds.numel().numpy().item() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel().numpy().item() == 0
            self.pos_gt_masks = paddle.empty_like(gt_masks)
        else:
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def masks(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return paddle.concat([self.pos_masks, self.neg_masks])

    def __nice__(self):
        data = self.info.copy()
        data['pos_masks'] = data.pop('pos_masks').shape
        data['neg_masks'] = data.pop('neg_masks').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_masks': self.pos_masks,
            'neg_masks': self.neg_masks,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }


class MaskPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, masks, gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            masks (torch.Tensor): Bounding boxes
            gt_masks (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = paddle.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = paddle.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = paddle.zeros([masks.shape[0]], dtype=paddle.int32)
        sampling_result = MaskSamplingResult(pos_inds, neg_inds, masks,
                                             gt_masks, assign_result, gt_flags)
        return sampling_result



BBOX_SAMPLERS = {
    "MaskPseudoSampler": MaskPseudoSampler
}


def build_sampler(cfg, **default_args):
    """Build a module from config dict.

    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.

    Returns:
        object: The constructed object.
    """
    registry = BBOX_SAMPLERS
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