import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ._utils import multi_apply, reduce_mean, bias_init_with_prob, normal_init
from .base_layer import ConvModule
from .semantic_fpn_wrapper import SemanticFPNWrapper
from .__loss import build_loss
from .mask_hungarian_assigner import build_assigner
from .mask_pseudo_sampler import build_sampler
# from mmdet.core import build_assigner, build_sampler
# from mmdet.models.builder import HEADS, build_loss, build_neck
# from mmdet.models.losses import accuracy



class ConvKernelHead(nn.Layer):

    def __init__(self,
                 num_proposals=100,
                 in_channels=256,
                 out_channels=256,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_seg_convs=1,
                 num_loc_convs=1,
                 att_dropout=False,
                 localization_fpn=None,
                 conv_kernel_size=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 semantic_fpn=True,
                 train_cfg=None,
                 num_classes=80,
                 xavier_init_kernel=False,
                 kernel_init_std=0.01,
                 use_binary=False,
                 proposal_feats_with_obj=False,
                 loss_mask=None,
                 loss_seg=None,
                 loss_cls=None,
                 loss_dice=None,
                 loss_rank=None,
                 assigner_config=None,
                 sampler_config=dict(type='MaskPseudoSampler'),
                 cfg_pos_weight=1,
                 feat_downsample_stride=1,
                 feat_refine_stride=1,
                 feat_refine=True,
                 with_embed=False,
                 feat_embed_only=False,
                 conv_normal_init=False,
                 mask_out_stride=4,
                 hard_target=False,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 cat_stuff_mask=False,
                 **kwargs):
        super(ConvKernelHead, self).__init__()
        self.num_proposals = num_proposals
        self.num_cls_fcs = num_cls_fcs
        self.train_cfg = train_cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.proposal_feats_with_obj = proposal_feats_with_obj
        self.sampling = False
        self.localization_fpn = SemanticFPNWrapper(**localization_fpn)
        self.semantic_fpn = semantic_fpn
        self.norm_cfg = norm_cfg
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.conv_kernel_size = conv_kernel_size
        self.xavier_init_kernel = xavier_init_kernel
        self.kernel_init_std = kernel_init_std
        self.feat_downsample_stride = feat_downsample_stride
        self.feat_refine_stride = feat_refine_stride
        self.conv_normal_init = conv_normal_init
        self.feat_refine = feat_refine
        self.with_embed = with_embed
        self.feat_embed_only = feat_embed_only
        self.num_loc_convs = num_loc_convs
        self.num_seg_convs = num_seg_convs
        self.use_binary = use_binary
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg
        self.cat_stuff_mask = cat_stuff_mask
        self.cfg_pos_weight = cfg_pos_weight

        if loss_mask is not None:
            self.loss_mask = build_loss(loss_mask)
        else:
            self.loss_mask = loss_mask

        if loss_dice is not None:
            self.loss_dice = build_loss(loss_dice)
        else:
            self.loss_dice = loss_dice

        if loss_seg is not None:
            self.loss_seg = build_loss(loss_seg)
        else:
            self.loss_seg = loss_seg
        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)
        else:
            self.loss_cls = loss_cls

        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank


            
        self.assigner = build_assigner(assigner_config)
        # use PseudoSampler when sampling is False
        # if self.sampling and hasattr(self.train_cfg, 'sampler'):
        #     sampler_cfg = self.train_cfg.sampler
        # else:
        #     sampler_cfg = dict(type='MaskPseudoSampler')
        self.sampler = build_sampler(sampler_config, context=self)
        
        self._init_layers()

    def _init_layers(self):
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_kernels = nn.Conv2D(
            self.out_channels,
            self.num_proposals,
            self.conv_kernel_size,
            padding=int(self.conv_kernel_size // 2),
            bias_attr=False)

        if self.semantic_fpn:
            if self.loss_seg.use_sigmoid:
                self.conv_seg = nn.Conv2D(self.out_channels, self.num_classes,
                                          1)
            else:
                self.conv_seg = nn.Conv2D(self.out_channels,
                                          self.num_classes + 1, 1)

        if self.feat_downsample_stride > 1 and self.feat_refine:
            self.ins_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,
                padding=1,
                norm_cfg=self.norm_cfg)
            self.seg_downsample = ConvModule(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.feat_refine_stride,
                padding=1,
                norm_cfg=self.norm_cfg)

        self.loc_convs = nn.LayerList()
        for i in range(self.num_loc_convs):
            self.loc_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

        self.seg_convs = nn.LayerList()
        for i in range(self.num_seg_convs):
            self.seg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.out_channels,
                    1,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        assert False, ""
        self.localization_fpn.init_weights()

        if self.feat_downsample_stride > 1 and self.conv_normal_init:
            # logger = get_root_logger()
            # logger.info('Initialize convs in KPN head by normal std 0.01')
            for conv in [self.loc_convs, self.seg_convs]:
                for m in conv.modules():
                    if isinstance(m, nn.Conv2D):
                        normal_init(m, std=0.01)

        if self.semantic_fpn:
            bias_seg = bias_init_with_prob(0.01)
            if self.loss_seg.use_sigmoid:
                normal_init(self.conv_seg, std=0.01, bias=bias_seg)
            else:
                normal_init(self.conv_seg, mean=0, std=0.01)
        if self.xavier_init_kernel:
            # logger = get_root_logger()
            # logger.info('Initialize kernels by xavier uniform')
            nn.init.xavier_uniform_(self.init_kernels.weight)
        else:
            # logger = get_root_logger()
            # logger.info(
            #     f'Initialize kernels by normal std: {self.kernel_init_std}')
            normal_init(self.init_kernels, mean=0, std=self.kernel_init_std)

    def _decode_init_proposals(self, img, img_metas):
        num_imgs = len(img_metas)

        localization_feats = self.localization_fpn(img) # [1] 是 [0] 卷积之后的结果, 通过原文中 semanticFPN 的结果
        if isinstance(localization_feats, list):
            loc_feats = localization_feats[0]
        else:
            loc_feats = localization_feats
        for conv in self.loc_convs:
            loc_feats = conv(loc_feats) # Conv+GN+ReLU
        if self.feat_downsample_stride > 1 and self.feat_refine: # False
            loc_feats = self.ins_downsample(loc_feats)
        mask_preds = self.init_kernels(loc_feats)   # i256 o100 k1 s1

        if self.semantic_fpn:
            if isinstance(localization_feats, list):
                semantic_feats = localization_feats[1]
            else:
                semantic_feats = localization_feats
            for conv in self.seg_convs:
                semantic_feats = conv(semantic_feats) # Conv+GN+ReLU
            if self.feat_downsample_stride > 1 and self.feat_refine: # False
                semantic_feats = self.seg_downsample(semantic_feats)
        else:
            semantic_feats = None

        if semantic_feats is not None:
            seg_preds = self.conv_seg(semantic_feats) # i256 o133 k1 s1
        else:
            seg_preds = None

        proposal_feats = self.init_kernels.weight.clone()
        proposal_feats = proposal_feats[None].expand([num_imgs,
                                                      *proposal_feats.shape]) # 0维重复一下需要这么搞吗

        if semantic_feats is not None:
            x_feats = semantic_feats + loc_feats # 语义分割与实例分割的结果相加
        else:
            x_feats = loc_feats

        if self.proposal_feats_with_obj: # True
            sigmoid_masks = F.sigmoid(mask_preds)
            nonzero_inds = sigmoid_masks > 0.5
            if self.use_binary: # True
                sigmoid_masks = nonzero_inds.cast("float32")
            else:
                sigmoid_masks = nonzero_inds.cast("float32") * sigmoid_masks
            obj_feats = paddle.einsum('bnhw,bchw->bnc', sigmoid_masks, x_feats) # bmm( sigmoid_masks.flatten(2), x_feats.flatten(2).transpose(1,2) )

        cls_scores = None

        if self.proposal_feats_with_obj: # True
            proposal_feats = proposal_feats + obj_feats.reshape(
                [num_imgs, self.num_proposals, self.out_channels, 1, 1])

        if self.cat_stuff_mask and not self.training: # False
            mask_preds = paddle.concat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], axis=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand(num_imgs,
                                                       *stuff_kernels.size())
            proposal_feats = paddle.concat([proposal_feats, stuff_kernels], axis=1)

        return proposal_feats, x_feats, mask_preds, cls_scores, seg_preds # mask_preds(实例分割的结果), cls_scores(None), seg_preds(语义分割结果)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,      # 实例分割部分
                      gt_labels,     # 实例分割类别
                      gt_sem_seg=None,   # 全景分割的掩码
                      gt_sem_cls=None):  # 全景分割的类别
        """Forward function in training stage."""
        num_imgs = len(img_metas) # batch size
        results = self._decode_init_proposals(img, img_metas)
        (proposal_feats, x_feats, mask_preds, cls_scores, seg_preds) = results
        if self.feat_downsample_stride > 1:
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=self.feat_downsample_stride, # 缩放到这个比例
                mode='bilinear',
                align_corners=False)
            if seg_preds is not None: # True
                scaled_seg_preds = F.interpolate(
                    seg_preds,
                    scale_factor=self.feat_downsample_stride,
                    mode='bilinear',
                    align_corners=False)
        else:
            scaled_mask_preds = mask_preds
            scaled_seg_preds = seg_preds

        if self.hard_target: # False
            gt_masks = [x.cast("bool").cast("float32") for x in gt_masks]
        else:
            gt_masks = gt_masks

        sampling_results = []
        if cls_scores is None: # True
            detached_cls_scores = [None] * num_imgs
        else:
            detached_cls_scores = cls_scores.detach()

        for i in range(num_imgs):
            assign_result = self.assigner.assign(scaled_mask_preds[i].detach(),
                                                 detached_cls_scores[i],
                                                 gt_masks[i], gt_labels[i],
                                                 img_metas[i])
            sampling_result = self.sampler.sample(assign_result,
                                                  scaled_mask_preds[i],
                                                  gt_masks[i])
            sampling_results.append(sampling_result)

        mask_targets = self.get_targets(
            sampling_results,
            gt_masks,
            # self.train_cfg,
            self.cfg_pos_weight,
            True,
            gt_sem_seg=gt_sem_seg,
            gt_sem_cls=gt_sem_cls)

        losses = self.loss(scaled_mask_preds, cls_scores, scaled_seg_preds,
                           proposal_feats, *mask_targets)

        if self.cat_stuff_mask and self.training:
            mask_preds = paddle.concat(
                [mask_preds, seg_preds[:, self.num_thing_classes:]], axis=1)
            stuff_kernels = self.conv_seg.weight[self.
                                                 num_thing_classes:].clone()
            stuff_kernels = stuff_kernels[None].expand([num_imgs,
                                                       *stuff_kernels.shape])
            proposal_feats = paddle.concat([proposal_feats, stuff_kernels], axis=1)

        return losses, proposal_feats, x_feats, mask_preds, cls_scores

    def loss(self,
             mask_pred,
             cls_scores,
             seg_preds,
             proposal_feats,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             seg_targets,
             reduction_override=None,
             **kwargs):
        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_preds = mask_pred.shape[0] * mask_pred.shape[1]

        if cls_scores is not None:
            num_pos = pos_inds.sum().cast("float32")
            avg_factor = reduce_mean(num_pos)
            assert mask_pred.shape[0] == cls_scores.shape[0]
            assert mask_pred.shape[1] == cls_scores.shape[1]
            losses['loss_rpn_cls'] = self.loss_cls(
                cls_scores.view(num_preds, -1),
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['rpn_pos_acc'] = accuracy(
                cls_scores.view(num_preds, -1)[pos_inds], labels[pos_inds])

        bool_pos_inds = pos_inds.cast("bool")
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        # do not perform bounding box regression for BG anymore.
        H, W = mask_pred.shape[-2:]
        if pos_inds.any():
            pos_mask_pred = mask_pred.reshape([num_preds, H, W])[bool_pos_inds]
            pos_mask_targets = mask_targets[bool_pos_inds]
            losses['loss_rpn_mask'] = self.loss_mask(pos_mask_pred,
                                                     pos_mask_targets)
            losses['loss_rpn_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets)

            if self.loss_rank is not None:
                batch_size = mask_pred.shape[0]
                # rank_target = mask_targets.new_full((batch_size, H, W),
                #                                     self.ignore_label,
                #                                     dtype="int64")
                rank_target = paddle.full((batch_size, H, W), self.ignore_label, dtype="int64")
                rank_inds = pos_inds.reshape([batch_size, -1]).nonzero(as_tuple=False)
                batch_mask_targets = mask_targets.reshape([batch_size, -1, H, W]).cast("bool")
                for i in range(batch_size):
                    curr_inds = (rank_inds[:, 0] == i)
                    curr_rank = rank_inds[:, 1][curr_inds]
                    for j in curr_rank:
                        rank_target[i][batch_mask_targets[i][j]] = j
                losses['loss_rpn_rank'] = self.loss_rank(
                    mask_pred, rank_target, ignore_index=self.ignore_label)

        else:
            losses['loss_rpn_mask'] = mask_pred.sum() * 0
            losses['loss_rpn_dice'] = mask_pred.sum() * 0
            if self.loss_rank is not None:
                losses['loss_rank'] = mask_pred.sum() * 0

        if seg_preds is not None:
            if self.loss_seg.use_sigmoid:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.reshape([
                    -1, cls_channel,
                    H * W]).transpose([0, 2, 1]).reshape([-1, cls_channel])
                flatten_seg_target = seg_targets.reshape([-1])
                num_dense_pos = (flatten_seg_target >= 0) & (
                    flatten_seg_target < bg_class_ind)
                num_dense_pos = num_dense_pos.sum().cast("float32").clip(min=1.0)
                losses['loss_rpn_seg'] = self.loss_seg(
                    flatten_seg,
                    flatten_seg_target,
                    avg_factor=num_dense_pos)
            else:
                cls_channel = seg_preds.shape[1]
                flatten_seg = seg_preds.reshape([-1, cls_channel, H * W]).transpose([0, 2, 1]).reshape([-1, cls_channel])
                flatten_seg_target = seg_targets.reshape([-1])
                losses['loss_rpn_seg'] = self.loss_seg(flatten_seg, flatten_seg_target)

        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                           pos_gt_mask, pos_gt_labels, gt_sem_seg, gt_sem_cls,
                           cfg_pos_weight=1):
        num_pos = pos_mask.shape[0]
        num_neg = neg_mask.shape[0]
        num_samples = num_pos + num_neg
        H, W = pos_mask.shape[-2:]
        
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        
        # labels = pos_mask.new_full((num_samples, ), self.num_classes, dtype="int64")
        # label_weights = pos_mask.new_zeros(num_samples)
        # mask_targets = pos_mask.new_zeros(num_samples, H, W)
        # mask_weights = pos_mask.new_zeros(num_samples, H, W)
        # seg_targets = pos_mask.new_full((H, W), self.num_classes, dtype="int64")
        
        labels = paddle.full((num_samples, ), self.num_classes, dtype="int64")
        label_weights = paddle.zeros([num_samples])
        mask_targets = paddle.zeros([num_samples, H, W])
        mask_weights = paddle.zeros([num_samples, H, W])
        seg_targets = paddle.full((H, W), self.num_classes, dtype="int64")
        

        if gt_sem_cls is not None and gt_sem_seg is not None:
            gt_sem_seg = gt_sem_seg.cast("bool")
            for sem_mask, sem_cls in zip(gt_sem_seg, gt_sem_cls):
                seg_targets[sem_mask] = sem_cls.cast("int64")

        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg_pos_weight <= 0 else cfg_pos_weight
            label_weights[pos_inds] = pos_weight
            mask_targets[pos_inds, ...] = pos_gt_mask
            mask_weights[pos_inds, ...] = 1
            for i in range(num_pos):
                seg_targets[pos_gt_mask[i].cast("bool")] = pos_gt_labels[i]

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        return labels, label_weights, mask_targets, mask_weights, seg_targets

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    cfg_pos_weight,
                    concat=True,
                    gt_sem_seg=None,
                    gt_sem_cls=None):
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        if gt_sem_seg is None:
            gt_sem_seg = [None] * 2
            gt_sem_cls = [None] * 2
        results = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg_pos_weight=cfg_pos_weight)
        (labels, label_weights, mask_targets, mask_weights, seg_targets) = results
        if concat:
            labels = paddle.concat(labels, 0)
            label_weights = paddle.concat(label_weights, 0)
            mask_targets = paddle.concat(mask_targets, 0)
            mask_weights = paddle.concat(mask_weights, 0)
            seg_targets = paddle.stack(seg_targets, 0)
        return labels, label_weights, mask_targets, mask_weights, seg_targets

    def simple_test_rpn(self, img, img_metas):
        """Forward function in testing stage."""
        return self._decode_init_proposals(img, img_metas)

    def forward_dummy(self, img, img_metas):
        """Dummy forward function.

        Used in flops calculation.
        """
        return self._decode_init_proposals(img, img_metas)


if __name__ == "__main__":

    model_config = dict(
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
    )

    model = ConvKernelHead(**model_config)
    x = [[2, 256, 200, 336], [2, 256, 100, 168], [2, 256, 50, 84], [2, 256, 25, 42]]
    x = [paddle.rand(x_sp) for x_sp in x]

    img_metas = [{}, {}]
    gt_mask = [(paddle.rand([4, 200, 336])>0.5).cast("float32") , 
            (paddle.rand([3, 200, 336])>0.5).cast("float32") ]

            
    gt_labels = [
        paddle.to_tensor([7, 7, 7, 7]),
        paddle.to_tensor([ 0, 27, 41])
    ]

    gt_sem_seg = [
        (paddle.rand([6, 200, 336])>0.5).cast("float32"), 
        (paddle.rand([3, 200, 336])>0.5).cast("float32") 
    ]

    gt_sem_cls = [
        paddle.to_tensor([ 100, 109, 116, 123, 125, 129]),
        paddle.to_tensor([ 112, 118, 131])
    ]

    y = model.forward_train(x, img_metas, gt_mask, gt_labels, gt_sem_seg, gt_sem_cls)
    print(y)