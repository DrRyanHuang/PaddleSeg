import paddle
import paddle.nn as nn
from .base_layer import ConvModule, BaseModule
from ._utils import normal_init
import math
# from mmcv.cnn.bricks.transformer import build_positional_encoding



class SinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(SinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.cast("int32")
        not_mask = 1 - mask  # logical_not
        y_embed = not_mask.cumsum(1, dtype="float32")
        x_embed = not_mask.cumsum(2, dtype="float32")
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = paddle.arange(
            self.num_feats, dtype=paddle.int32)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.shape
        pos_x = paddle.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            axis=4).reshape([B, H, W, -1])
        pos_y = paddle.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            axis=4).reshape([B, H, W, -1])
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        return pos

    # def __repr__(self):
    #     """str: a string that describes the module"""
    #     repr_str = self.__class__.__name__
    #     repr_str += f'(num_feats={self.num_feats}, '
    #     repr_str += f'temperature={self.temperature}, '
    #     repr_str += f'normalize={self.normalize}, '
    #     repr_str += f'scale={self.scale}, '
    #     repr_str += f'eps={self.eps})'
    #     return repr_str


if __name__ == "__main__":
    
    spe_config = dict(
        num_feats=128, 
        normalize=True
    )
    model = SinePositionalEncoding(**spe_config)
    x = paddle.rand([2, 34, 24])
    positional_encoding = model(x)




class SemanticFPNWrapper(nn.Layer):
    """Implementation of Semantic FPN used in Panoptic FPN.

    Args:
        in_channels ([type]): [description]
        feat_channels ([type]): [description]
        out_channels ([type]): [description]
        start_level ([type]): [description]
        end_level ([type]): [description]
        cat_coors (bool, optional): [description]. Defaults to False.
        fuse_by_cat (bool, optional): [description]. Defaults to False.
        conv_cfg ([type], optional): [description]. Defaults to None.
        norm_cfg ([type], optional): [description]. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 start_level,
                 end_level,
                 cat_coors=False,
                 positional_encoding=None,
                 cat_coors_level=3,
                 fuse_by_cat=False,
                 return_list=False,
                 upsample_times=3,
                 with_pred=True,
                 num_aux_convs=0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 out_act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticFPNWrapper, self).__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cat_coors = cat_coors
        self.cat_coors_level = cat_coors_level
        self.fuse_by_cat = fuse_by_cat
        self.return_list = return_list
        self.upsample_times = upsample_times
        self.with_pred = with_pred
        if positional_encoding is not None:
            # self.positional_encoding = build_positional_encoding(
            #     positional_encoding)
            self.positional_encoding = SinePositionalEncoding(**positional_encoding)
        else:
            self.positional_encoding = None

        self.convs_all_levels = nn.LayerList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                if i == self.cat_coors_level and self.cat_coors:
                    chn = self.in_channels + 2
                else:
                    chn = self.in_channels
                if upsample_times == self.end_level - i:
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_sublayer('conv' + str(i), one_conv)
                else:
                    for i in range(self.end_level - upsample_times):
                        one_conv = ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            stride=2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            inplace=False)
                        convs_per_level.add_sublayer('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.cat_coors_level and self.cat_coors:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_sublayer('conv' + str(j), one_conv)
                    if j < upsample_times - (self.end_level - i):
                        one_upsample = nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
                        convs_per_level.add_sublayer('upsample' + str(j),
                                                   one_upsample)
                    continue

                one_conv = ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                convs_per_level.add_sublayer('conv' + str(j), one_conv)
                if j < upsample_times - (self.end_level - i):
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_sublayer('upsample' + str(j),
                                               one_upsample)

            self.convs_all_levels.append(convs_per_level)

        if fuse_by_cat:
            in_channels = self.feat_channels * len(self.convs_all_levels)
        else:
            in_channels = self.feat_channels

        if self.with_pred:
            self.conv_pred = ConvModule(
                in_channels,
                self.out_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                act_cfg=out_act_cfg,
                norm_cfg=self.norm_cfg)

        self.num_aux_convs = num_aux_convs
        self.aux_convs = nn.LayerList()
        for i in range(num_aux_convs):
            self.aux_convs.append(
                ConvModule(
                    in_channels,
                    self.out_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    act_cfg=out_act_cfg,
                    norm_cfg=self.norm_cfg))

    def init_weights(self):
        # logger = get_root_logger()
        # logger.info('Use normal intialization for semantic FPN')
        assert False
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                normal_init(m, std=0.01)

    def generate_coord(self, input_feat):
        x_range = paddle.linspace(-1, 1, input_feat.shape[-1])
        y_range = paddle.linspace(-1, 1, input_feat.shape[-2])
        y, x = paddle.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = paddle.concat([x, y], 1)
        return coord_feat

    def forward(self, inputs):
        mlvl_feats = []
        for i in range(self.start_level, self.end_level + 1):
            input_p = inputs[i]
            if i == self.cat_coors_level:
                if self.positional_encoding is not None:
                    ignore_mask = paddle.zeros(
                        (input_p.shape[0], input_p.shape[-2],
                         input_p.shape[-1]),
                        dtype="bool")
                    positional_encoding = self.positional_encoding(ignore_mask)
                    input_p = input_p + positional_encoding      # 给最后的 feature 加位置编码
                if self.cat_coors:
                    coord_feat = self.generate_coord(input_p)
                    input_p = paddle.concat([input_p, coord_feat], 1)

            mlvl_feats.append(self.convs_all_levels[i](input_p)) # Conv+GN+ReLU, (若有上采样则上采样后, 在Conv)

        if self.fuse_by_cat:
            feature_add_all_level = paddle.concat(mlvl_feats, dim=1)
        else:
            feature_add_all_level = sum(mlvl_feats)

        if self.with_pred:
            out = self.conv_pred(feature_add_all_level)  # 求和后再过一次卷积
        else:
            out = feature_add_all_level

        if self.num_aux_convs > 0:
            outs = [out]
            for conv in self.aux_convs:
                outs.append(conv(feature_add_all_level))
            return outs

        if self.return_list:
            return [out]
        else:
            return out


if __name__ == "__main__":

    localization_fpn=dict(
        in_channels=256,
        feat_channels=256,
        out_channels=256,
        start_level=0,
        end_level=3,
        upsample_times=2,
        positional_encoding=dict(
            num_feats=128, normalize=True),
        cat_coors=False,
        cat_coors_level=3,
        fuse_by_cat=False,
        return_list=False,
        num_aux_convs=1,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    )


    model = SemanticFPNWrapper(**localization_fpn)

    x = [[2, 256, 304, 200], [2, 256, 152, 100], [2, 256, 76, 50], [2, 256, 38, 25]]
    x = [paddle.rand(sp) for sp in x]
    y = model(x)