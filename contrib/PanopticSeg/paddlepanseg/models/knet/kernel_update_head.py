import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# from mmcv.cnn.bricks.transformer import build_transformer_layer

from base_layer import build_activation_layer, build_norm_layer, ConvModule, FFN, MultiheadAttention
from _utils import multi_apply, reduce_mean, bias_init_with_prob
from __loss import build_loss, accuracy

# from kernel_update_head import KernelUpdator
# from mmdet.models.losses import accuracy
# from mmdet.utils import get_root_logger



class KernelUpdator(nn.Layer):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=3,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels)
        if self.gate_norm_act:
            self.gate_norm = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.input_norm_out = build_norm_layer(norm_cfg, self.feat_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels)
        self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, update_feature, input_feature): # x_feat, Dynamic Kernel
        update_feature = update_feature.reshape([-1, self.in_channels])
        num_proposals = update_feature.shape[0]
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[:, :self.num_params_in].reshape(
            [-1, self.feat_channels])
        param_out = parameters[:, -self.num_params_out:].reshape(
            [-1, self.feat_channels])

        input_feats = self.input_layer(
            input_feature.reshape([num_proposals, -1, self.feat_channels]))
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in.unsqueeze(-2) # F^G in paper
        if self.gate_norm_act: # False
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats)) # G^K
        update_gate = self.norm_in(self.update_gate(gate_feats))     # G^F
        if self.gate_sigmoid: # True
            input_gate = F.sigmoid(input_gate)
            update_gate = F.sigmoid(update_gate)
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out: # False
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out.unsqueeze(
            -2) + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features




if __name__ == "__main__":
    
    update_feature = paddle.rand([2, 153, 256])
    input_feature = paddle.rand([2, 153, 1, 256])

    _config = dict(
        # type='KernelUpdator',
        in_channels=256,
        feat_channels=256,
        out_channels=256,
        input_feat_shape=3,
        act_cfg=dict(type='ReLU', inplace=True),
        norm_cfg=dict(type='LN')
    )

    model = KernelUpdator(**_config)
    y = model(update_feature, input_feature)





class KernelUpdateHead(nn.Layer):

    def __init__(self,
                 num_classes=80,
                 num_ffn_fcs=2,
                 num_heads=8,
                 num_cls_fcs=1,
                 num_mask_fcs=3,
                 feedforward_channels=2048,
                 in_channels=256,
                 out_channels=256,
                 dropout=0.0,
                 mask_thr=0.5,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_act_cfg=dict(type='ReLU', inplace=True),
                 conv_kernel_size=3,
                 feat_transform_cfg=None,
                 hard_mask_thr=0.5,
                 kernel_init=False,
                 with_ffn=True,
                 mask_out_stride=4,
                 relative_coors=False,
                 relative_coors_off=False,
                 feat_gather_stride=1,
                 mask_transform_stride=1,
                 mask_upsample_stride=1,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 kernel_updator_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=1,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 loss_rank=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 loss_dice=dict(type='DiceLoss', loss_weight=3.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0)):
        super(KernelUpdateHead, self).__init__()
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.mask_upsample_stride = mask_upsample_stride

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg

        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size**2)[1]

        # self.kernel_update_conv = build_transformer_layer(kernel_updator_cfg)
        if "type" in kernel_updator_cfg:
            kernel_updator_cfg.pop("type")
        self.kernel_update_conv = KernelUpdator(**kernel_updator_cfg)

        if feat_transform_cfg is not None:
            kernel_size = feat_transform_cfg.pop('kernel_size', 1)
            self.feat_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
        else:
            self.feat_transform = None

        if self.with_ffn:
            self.ffn = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.LayerList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias_attr=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))

        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.mask_fcs = nn.LayerList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias_attr=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        if self.kernel_init:
            # logger = get_root_logger()
            # logger.info(
            #     'mask kernel in mask head is normal initialized by std 0.01')
            nn.init.normal_(self.fc_mask.weight, mean=0, std=0.01)

    def forward(self,
                x,                     # semanticFPN的输出
                proposal_feat,         # Learned/Dynamic Kernel
                mask_preds,            # Mask Prediction
                prev_cls_score=None,
                mask_shape=None,
                img_metas=None):

        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W: # False
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = F.sigmoid(gather_mask)
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.cast("float32") # Only 1 or 0  |  unique: [0., 1.]

        # einsum is faster than bmm by 30%
        x_feat = paddle.einsum('bnhw,bchw->bnc', sigmoid_masks, x) # F^K

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape([N, num_proposals,
                                              self.in_channels, -1]).transpose([0, 1, 3, 2])
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)  # tilde K in paper

        # -------------------------- Kernel Interaction --------------------------
        
        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape([N, num_proposals, -1]).transpose([1, 0, 2])
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.transpose([1, 0, 2])

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape([N, num_proposals, -1, self.in_channels])

        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        cls_score = self.fc_cls(cls_feat).reshape([N, num_proposals, -1])
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).transpose([0, 1, 3, 2])

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape([N, num_proposals, C,
                                       self.conv_kernel_size,
                                       self.conv_kernel_size])
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = paddle.concat(new_mask_preds, axis=0)
        new_mask_preds = new_mask_preds.reshape([N, num_proposals, H, W])
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')

        return cls_score, new_mask_preds, obj_feat.transpose([0, 1, 3, 2]).reshape(
            [N, num_proposals, self.in_channels, 
             self.conv_kernel_size, self.conv_kernel_size])

    # @force_fp32(apply_to=('cls_score', 'mask_pred'))
    def loss(self,
             object_feats,
             cls_score,
             mask_pred,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):

        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().cast("float32")
        avg_factor = reduce_mean(num_pos).clip(min=1.0)

        num_preds = mask_pred.shape[0] * mask_pred.shape[1]
        assert mask_pred.shape[0] == cls_score.shape[0]
        assert mask_pred.shape[1] == cls_score.shape[1]

        if cls_score is not None:
            if cls_score.numel().numpy().item() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score.reshape([num_preds, -1]),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(
                    cls_score.reshape([num_preds, -1])[pos_inds], labels[pos_inds])
        if mask_pred is not None:
            bool_pos_inds = pos_inds.cast("bool")
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            H, W = mask_pred.shape[-2:]
            if pos_inds.any():
                pos_mask_pred = mask_pred.reshape([num_preds, H, W])[bool_pos_inds]
                pos_mask_targets = mask_targets[bool_pos_inds]
                losses['loss_mask'] = self.loss_mask(pos_mask_pred,
                                                     pos_mask_targets)
                losses['loss_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets)

                if self.loss_rank is not None:
                    batch_size = mask_pred.shape[0]
                    # rank_target = paddle.full((batch_size, H, W),
                    #                 self.ignore_label, dtype="int64").cast(mask_targets.dtype)
                    rank_target = [paddle.full((H, W),
                                    self.ignore_label, dtype="int64").cast(mask_targets.dtype) for _ in range(batch_size)]
                    
                    
                    rank_inds = pos_inds.reshape([batch_size, -1]).nonzero(as_tuple=False)
                    batch_mask_targets = mask_targets.reshape([batch_size, -1, H, W]).cast("bool")
                    for i in range(batch_size):
                        curr_inds = (rank_inds[:, 0] == i)
                        curr_rank = rank_inds[:, 1][curr_inds]
                        for j in curr_rank:
                            rank_target[i][batch_mask_targets[i][j]] = j.cast("float32")
                            
                    rank_target = paddle.stack(rank_target, axis=0)
                    losses['loss_rank'] = self.loss_rank(
                        mask_pred, rank_target, ignore_index=self.ignore_label)
            else:
                losses['loss_mask'] = mask_pred.sum() * 0
                losses['loss_dice'] = mask_pred.sum() * 0
                if self.loss_rank is not None:
                    losses['loss_rank'] = mask_pred.sum() * 0

        return losses

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                           pos_gt_mask, pos_gt_labels, gt_sem_seg, gt_sem_cls,
                           cfg):

        num_pos = pos_mask.shape[0]
        num_neg = neg_mask.shape[0]
        num_samples = num_pos + num_neg
        H, W = pos_mask.shape[-2:]
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = paddle.full((num_samples, ), self.num_classes, dtype="int64")
        # label_weights = pos_mask.new_zeros((num_samples, self.num_classes))
        # mask_targets = pos_mask.new_zeros(num_samples, H, W)
        # mask_weights = pos_mask.new_zeros(num_samples, H, W)

        label_weights = paddle.zeros([num_samples, self.num_classes]).cast(pos_mask.dtype)
        mask_targets = paddle.zeros([num_samples, H, W]).cast(pos_mask.dtype)
        mask_weights = paddle.zeros([num_samples, H, W]).cast(pos_mask.dtype)
        
        
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg['pos_weight'] <= 0 else cfg['pos_weight']
            label_weights[pos_inds] = pos_weight
            pos_mask_targets = pos_gt_mask
            mask_targets[pos_inds, ...] = pos_mask_targets
            mask_weights[pos_inds, ...] = 1

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if gt_sem_cls is not None and gt_sem_seg is not None:
            # sem_labels = pos_mask.new_full((self.num_stuff_classes, ),
            #                                self.num_classes,
            #                                dtype="int64")
            # sem_targets = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            # sem_weights = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            
            sem_labels = paddle.full((self.num_stuff_classes,), self.num_classes, dtype="int64")
            sem_targets = paddle.zeros([self.num_stuff_classes, H, W]).cast(pos_mask.dtype)
            sem_weights = paddle.zeros([self.num_stuff_classes, H, W]).cast(pos_mask.dtype)
            
            sem_stuff_weights = paddle.eye(self.num_stuff_classes)
            # sem_thing_weights = pos_mask.new_zeros(
            #     (self.num_stuff_classes, self.num_thing_classes))
            sem_thing_weights = paddle.zeros(
                (self.num_stuff_classes, self.num_thing_classes)).cast(pos_mask.dtype)
            sem_label_weights = paddle.concat(
                [sem_thing_weights, sem_stuff_weights], axis=-1)
            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_thing_classes
                sem_inds = sem_inds.cast("int64")
                sem_labels[sem_inds] = gt_sem_cls.cast("int64")
                sem_targets[sem_inds] = gt_sem_seg
                sem_weights[sem_inds] = 1

            label_weights[:, self.num_thing_classes:] = 0
            labels = paddle.concat([labels, sem_labels])
            label_weights = paddle.concat([label_weights, sem_label_weights])
            mask_targets = paddle.concat([mask_targets, sem_targets])
            mask_weights = paddle.concat([mask_weights, sem_weights])

        return labels, label_weights, mask_targets, mask_weights

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    gt_labels,
                    rcnn_train_cfg,
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

        labels, label_weights, mask_targets, mask_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg=rcnn_train_cfg)
        if concat:
            labels = paddle.concat(labels, 0)
            label_weights = paddle.concat(label_weights, 0)
            mask_targets = paddle.concat(mask_targets, 0)
            mask_weights = paddle.concat(mask_weights, 0)
        return labels, label_weights, mask_targets, mask_weights

    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

    def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img,
                      test_cfg, img_meta):
        # resize mask predictions back
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        seg_masks = seg_masks > test_cfg.mask_thr
        bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img,
                                                    scores_per_img)
        return bbox_result, segm_result

    def segm2result(self, mask_preds, det_labels, cls_scores):
        num_classes = self.num_classes
        bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        mask_preds = mask_preds.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bbox_result = [bboxes[det_labels == i, :] for i in range(num_classes)]
        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
        return bbox_result, segm_result



if __name__ == "__main__":
    
    config = dict(
        # type='KernelUpdateHead',
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
            # type='KernelUpdator',
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

    model = KernelUpdateHead(**config)


    x = paddle.ones([2, 256, 96, 144])
    proposal_feat = paddle.ones([2, 153, 256, 1, 1])
    mask_preds = paddle.ones([2, 153, 96, 144])
    prev_cls_score = None
    mask_shape = None
    img_metas = [{}, {}]

    cls_score, mask_preds, object_feats = model(x, proposal_feat, 
                                                mask_preds, prev_cls_score, 
                                                mask_shape, img_metas)