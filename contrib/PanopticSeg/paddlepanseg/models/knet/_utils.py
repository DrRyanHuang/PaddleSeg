# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
import paddle.distributed as dist
import numpy as np
import paddle
# from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor



def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init



def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)



def sem2ins_masks(gt_sem_seg,
                  num_thing_classes=80):
    """Convert semantic segmentation mask to binary masks

    Args:
        gt_sem_seg (torch.Tensor): Semantic masks to be converted.
            [0, num_thing_classes-1] is the classes of things,
            [num_thing_classes:] is the classes of stuff.
        num_thing_classes (int, optional): Number of thing classes.
            Defaults to 80.

    Returns:
        tuple[torch.Tensor]: (mask_labels, bin_masks).
            Mask labels and binary masks of stuff classes.
    """
    # gt_sem_seg is zero-started, where zero indicates the first class
    # since mmdet>=2.17.0, see more discussion in
    # https://mmdetection.readthedocs.io/en/latest/conventions.html#coco-panoptic-dataset  # noqa
    classes = paddle.unique(gt_sem_seg)
    # classes ranges from 0 - N-1, where the class IDs in
    # [0, num_thing_classes - 1] are IDs of thing classes
    masks = []
    labels = []

    for i in classes:
        # skip ignore class 255 and "thing classes" in semantic seg
        if i == 255 or i < num_thing_classes:
            continue
        labels.append(i)
        masks.append(gt_sem_seg == i)

    if len(labels) > 0:
        labels = paddle.stack(labels)
        masks = paddle.concat(masks)
    else:
        labels = gt_sem_seg.new_zeros(size=[0])
        masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return labels.cast("int64"), masks.cast("float32")
