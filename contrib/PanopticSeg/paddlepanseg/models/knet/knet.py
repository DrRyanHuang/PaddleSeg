import paddle
import paddle.nn.functional as F

from two_stage import TwoStageDetector
from _utils import sem2ins_masks

from _fpn import FPN
from _resnet import ResNet
from kernel_head import ConvKernelHead
from kernel_iter_head import KernelIterHead

import matplotlib.pyplot as plt

class KNet(TwoStageDetector):

    def __init__(self,
                 *args,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,  # what??
                 thing_label_in_seg=0,
                 **kwargs):
        super(KNet, self).__init__(*args, **kwargs)
        
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        
        resnet_config = kwargs.get('backbone', None)
        fpn_config = kwargs.get('neck', None)

        self.backbone = ResNet(**resnet_config)
        self.neck = FPN(**fpn_config)
        
        rpn_head = kwargs.get('rpn_head', None)
        roi_head = kwargs.get('roi_head', None)
        train_cfg = kwargs.get('train_cfg', None)
        test_cfg = kwargs.get('test_cfg', None)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg["rpn"] if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg["rpn"])
            self.rpn_head = ConvKernelHead(**rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg["rcnn"] if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg["rcnn"])
            # roi_head.pretrained = pretrained
            self.roi_head = KernelIterHead(**roi_head)
        
        assert self.with_rpn, 'KNet does not support external proposals'

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes=None,   # Boxes 的坐标
                      gt_labels=None,   # Boxes 的类别
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      gt_semantic_seg=None,
                      **kwargs):

        super(TwoStageDetector, self).forward_train(img, img_metas)
        assert proposals is None, 'KNet does not support' \
                                  ' external proposals'
        assert gt_masks is not None

        # gt_masks and gt_semantic_seg are not padded when forming batch
        gt_masks_tensor = []
        gt_sem_seg = []
        gt_sem_cls = []
        # batch_input_shape shoud be the same across images
        pad_H, pad_W = img_metas[0]['batch_input_shape']
        assign_H = pad_H // self.mask_assign_stride
        assign_W = pad_W // self.mask_assign_stride

        for i, gt_mask in enumerate(gt_masks):
            mask_tensor = gt_mask.to_tensor(paddle.float32)
            if gt_mask.width != pad_W or gt_mask.height != pad_H:
                pad_wh = (0, pad_W - gt_mask.width, 0, pad_H - gt_mask.height)
                mask_tensor = F.pad(mask_tensor, pad_wh, value=0)
            
            # 语义分割部分
            if gt_semantic_seg is not None:
                # gt_semantic seg is padded by 255 and
                # zero indicating the first class
                sem_labels, sem_seg = sem2ins_masks(
                    gt_semantic_seg[i],
                    num_thing_classes=self.num_thing_classes)
                if sem_seg.shape[0] == 0:
                    gt_sem_seg.append(
                        mask_tensor.new_zeros(
                            (mask_tensor.size(0), assign_H, assign_W)))
                else:
                    gt_sem_seg.append(
                        F.interpolate(
                            sem_seg[None], (assign_H, assign_W),
                            mode='bilinear',
                            align_corners=False)[0])
                gt_sem_cls.append(sem_labels)

            else:
                gt_sem_seg = None
                gt_sem_cls = None

            # 实例分割部分
            if mask_tensor.shape[0] == 0:
                gt_masks_tensor.append(
                    mask_tensor.new_zeros(
                        (mask_tensor.size(0), assign_H, assign_W)))
            else:
                gt_masks_tensor.append(
                    F.interpolate(
                        mask_tensor[None], (assign_H, assign_W),
                        mode='bilinear',
                        align_corners=False)[0])

        gt_masks = gt_masks_tensor
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.forward_train(x, img_metas, gt_masks,   # gt_masks 是 instance 的掩码
                                                  gt_labels, gt_sem_seg,    # gt_sem_seg 语义分割掩码
                                                  gt_sem_cls)               # gt_sem_cls 掩码的类别
        (rpn_losses, proposal_feats, x_feats, mask_preds,
         cls_scores) = rpn_results

        losses = self.roi_head.forward_train(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,                         # None
            img_metas,
            gt_masks,                           # 实例的 mask
            gt_labels,                          # 检测Box的类别
            gt_bboxes_ignore=gt_bboxes_ignore,  # None
            gt_bboxes=gt_bboxes,                # 检测的 Box
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls,
            imgs_whwh=None)

        losses.update(rpn_losses)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        rpn_results = self.rpn_head.simple_test_rpn(x, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        segm_results = self.roi_head.simple_test(
            x_feats,
            proposal_feats,
            mask_preds,
            cls_scores,
            img_metas,
            imgs_whwh=None,
            rescale=rescale)
        return segm_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        # backbone
        x = self.extract_feat(img)
        # rpn
        num_imgs = len(img)
        dummy_img_metas = [
            dict(img_shape=(800, 1333, 3)) for _ in range(num_imgs)
        ]
        rpn_results = self.rpn_head.simple_test_rpn(x, dummy_img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores,
         seg_preds) = rpn_results
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x_feats, proposal_feats,
                                               dummy_img_metas)
        return roi_outs



model_config = dict(
    # type='KNet',
    backbone=dict(
        # type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        # type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        # type='ConvKernelHead',
        num_classes=133,
        cat_stuff_mask=True,
        conv_kernel_size=1,
        feat_downsample_stride=2,
        feat_refine_stride=1,
        feat_refine=False,
        use_binary=True,
        num_loc_convs=1,
        num_seg_convs=1,
        conv_normal_init=True,
        localization_fpn=dict(
            # type='SemanticFPNWrapper',
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            upsample_times=2,
            positional_encoding=dict(
                # type='SinePositionalEncoding', 
                num_feats=128, normalize=True),
            cat_coors=False,
            cat_coors_level=3,
            fuse_by_cat=False,
            return_list=False,
            num_aux_convs=1,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        num_proposals=100,
        proposal_feats_with_obj=True,
        xavier_init_kernel=False,
        kernel_init_std=1,
        num_cls_fcs=1,
        in_channels=256,
        feat_transform_cfg=None,
        loss_rank=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
        loss_seg=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=4.0),
        assigner_config=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
            mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)
            ),
        cfg_pos_weight=1,
        ),
    roi_head=dict(
        # type='KernelIterHead',
        do_panoptic=True,
        num_stages=3,
        stage_loss_weights=[1, 1, 1],
        proposal_feature_channel=256,
        mask_head=[
            dict(
                type='KernelUpdateHead',
                num_classes=133,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=1,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_rank=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)),
            dict(
                type='KernelUpdateHead',
                num_classes=133,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=1,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_rank=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)),
            dict(
                type='KernelUpdateHead',
                num_classes=133,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_mask_fcs=1,
                feedforward_channels=2048,
                in_channels=256,
                out_channels=256,
                dropout=0.0,
                mask_thr=0.5,
                conv_kernel_size=1,
                mask_upsample_stride=2,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                with_ffn=True,
                feat_transform_cfg=dict(
                    conv_cfg=dict(type='Conv2d'), act_cfg=None),
                kernel_updator_cfg=dict(
                    type='KernelUpdator',
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    input_feat_shape=3,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_rank=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                mask_cost=dict(type='MaskCost', weight=1.0, pred_act=True)),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(type='DiceCost', weight=4.0, pred_act=True),
                    mask_cost=dict(type='MaskCost', weight=1.0,
                                   pred_act=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1)
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=100,
            mask_thr=0.5,
            stuff_score_thr=0.05,
            merge_stuff_thing=dict(
                overlap_thr=0.6,
                iou_thr=0.5,
                stuff_max_area=4096,
                instance_score_thr=0.3)))
    )

knet = KNet(**model_config)


x = paddle.rand([2, 3, 800, 1344])
img_metas = [{}, {}]
gt_bboxes = [
    paddle.to_tensor([[262.1540, 490.1866, 546.9400, 678.7200],
                      [194.2580,  35.8213, 731.7680, 382.7227]]),
    paddle.to_tensor([[  99.9750,  189.6450, 1270.5156,  610.6152]])
]
gt_labels = [
    paddle.to_tensor([15, 15]),
    paddle.to_tensor([4])
]
gt_bboxes_ignore = None
proposals=None

gt_masks=None

gt_semantic_seg = paddle.randint(0, 255, [2, 1, 800, 1344],dtype="int32")


knet.forward_train()