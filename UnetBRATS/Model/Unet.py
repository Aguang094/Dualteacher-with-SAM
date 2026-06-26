import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from timm.models.layers import DropPath, trunc_normal_

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.data_format = data_format
        
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, x.shape[-4:], self.weight, self.bias, self.eps)
        
        # channels_first: (B, C, D, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()

def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)
    if name.lower() == "bn":
        return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01)
    if name.lower() == "1b":
        return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
    if name.lower() == "2b":
        return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
    if name.lower() == "3b":
        return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01)
    return nn.Identity()

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 norm=None, act=None,
                 conv_type='3d', conv_init='he_normal', norm_init=1.0):
        super().__init__()

        if conv_type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

        # Init weights
        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        
        if bias and self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        if norm is not None and hasattr(self.norm, 'weight') and self.norm.weight is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
        (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)


class opmoudle(nn.Module):
    """
    Improved Fusion Module: Uses learnable convolution instead of scalar weighting.
    Allows the model to learn spatial dependencies for fusion.
    """
    def __init__(self, n_classes=2):
        super(opmoudle, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(n_classes * 2, n_classes, kernel_size=1, bias=False),
            nn.BatchNorm3d(n_classes),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_classes, n_classes, kernel_size=1, bias=True) # Final projection
        )
        
        nn.init.constant_(self.fusion_conv[-1].weight, 0.0)
        nn.init.constant_(self.fusion_conv[-1].bias, 0.0)

    def forward(self, or_pre, mm_pre):
        """
        or_pre: UNet Final Output (Logits) [B, C, D, H, W]
        mm_pre: kMaX Reconstructed Output (Logits) [B, C, D, H, W]
        """
        cat_feat = torch.cat([or_pre, mm_pre], dim=1) 
        
        delta = self.fusion_conv(cat_feat)
        
        op_pre = or_pre + delta
        return op_pre

class AttentionOperation3D(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()
        self._batch_norm_similarity = get_norm('2b', num_heads) 
        self._batch_norm_retrieved_value = get_norm('1b', channels_v)

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        
        retrieved_value = torch.einsum('bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value


class kMaXPredictor3D(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=133 + 1, num_queries=16):
        super().__init__()
        self.num_queries = num_queries
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5,
                                                   groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='bn', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=1, bias=False, norm='bn',
                                                   act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=1, bias=True, norm='bn', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(in_channel_query, in_channel_query, kernel_size=1, bias=False, norm='1b', 
                                             act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(in_channel_query, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('3b', channels=self.num_queries)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        
        mask_logits = torch.einsum('bcdhw,bcn->bndhw', pixel_space_normalized_feature, cluster_mask_kernel)
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits)
        
        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
            'pixel_feature': pixel_space_normalized_feature
        }


class kMaXTransformerLayer3D(nn.Module):
    def __init__(
            self,
            num_classes=133,
            num_queries=16, 
            in_channel_pixel=2048,
            in_channel_query=16,
            base_filters=128,
            num_heads=8,
            bottleneck_expansion=2,
            key_expansion=1,
            value_expansion=2,
            drop_path_prob=0.0
    ):
        super().__init__()

        self._num_classes = num_classes
        self._num_heads = num_heads
        self._in_channel_query = in_channel_query
        self._num_queries = num_queries
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))
        
        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5
        self._query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='1b', act='gelu', conv_type='1d')

        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='bn', act='gelu')

        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='1b', act=None, conv_type='1d')
        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                       norm='bn', act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation3D(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                      norm='1b', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, in_channel_query, kernel_size=1, bias=False,
                                              norm='1b', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(in_channel_query, in_channel_query, kernel_size=1, bias=False,
                                          norm='1b', act=None, conv_type='1d', norm_init=0.0)

        self._predictor = kMaXPredictor3D(
            in_channel_pixel=self._bottleneck_channels,
            in_channel_query=self._bottleneck_channels, 
            num_classes=num_classes, 
            num_queries=self._num_queries
        )
        
        self._kmeans_query_batch_norm_retrieved_value = get_norm('1b', self._total_value_depth)
        self._kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                              norm='1b', act=None, conv_type='1d', norm_init=0.0)

    def forward(self, pixel_feature, query_feature):
        N, C, D, H, W = pixel_feature.shape
        _, Q, L = query_feature.shape

        if Q != self._in_channel_query:
            if Q > self._in_channel_query:
                query_feature = query_feature[:, :self._in_channel_query, :]
            else:
                padding = torch.zeros(N, self._in_channel_query - Q, L, device=query_feature.device)
                query_feature = torch.cat([query_feature, padding], dim=1)

        pixel_space = self._pixel_conv1_bn_act(pixel_feature) 
        query_space = self._query_conv1_bn_act(query_feature) 

        pixel_value = self._pixel_v_conv_bn(pixel_space) 
        pixel_value = pixel_value.reshape(N, self._total_value_depth, D * H * W)
        
        prediction_result = self._predictor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)

        with torch.no_grad():
            clustering_result = prediction_result['mask_logits'].flatten(2).detach() 
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result,
                                                 memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)

        with autocast(enabled=False):
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float()) 

        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
                                                [self._total_key_depth, self._total_key_depth, self._total_value_depth],
                                                dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)
        
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)
        
        # FFN
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, prediction_result

def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.InstanceNorm3d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        for m in self.children():
            init_weights(m)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up= nn.ConvTranspose3d(in_channels=in_size,
                                    out_channels=in_size,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1
                                    )
        self.bn1 = nn.InstanceNorm3d(num_features=in_size)
        self.relu = nn.ReLU(inplace=True)

        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        outputs2 = self.bn1(outputs2)
        outputs2 = self.relu(outputs2)
        
        if inputs1.size() != outputs2.size():
            diff_d = inputs1.size(2) - outputs2.size(2)
            diff_h = inputs1.size(3) - outputs2.size(3)
            diff_w = inputs1.size(4) - outputs2.size(4)
            outputs2 = F.pad(outputs2, [diff_w // 2, diff_w - diff_w // 2,
                                        diff_h // 2, diff_h - diff_h // 2,
                                        diff_d // 2, diff_d - diff_d // 2])
        
        return self.conv(torch.cat([inputs1, outputs2], 1))


class UNet_kMaX(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True, mun_pro=16, 
                 num_queries=16, query_dim=64):
        super(UNet_kMaX, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.mun_pro = mun_pro
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.n_classes = n_classes

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # --- 1. Encoder ---
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # --- 2. Decoder ---
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm) 
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm) 
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm) 
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm) 

        self.query_feat = nn.Parameter(torch.zeros(1, self.query_dim, self.num_queries))
        trunc_normal_(self.query_feat, std=0.02)

        # Layer 1
        self.kmax_layer1 = kMaXTransformerLayer3D(
            num_classes=n_classes,
            in_channel_pixel=filters[2], 
            in_channel_query=self.query_dim,
            num_queries=self.num_queries,
            base_filters=64,     
            num_heads=4
        )

        # Layer 2
        self.kmax_layer2 = kMaXTransformerLayer3D(
            num_classes=n_classes,
            in_channel_pixel=filters[1], 
            in_channel_query=self.query_dim,
            num_queries=self.num_queries,
            base_filters=32,
            num_heads=4
        )

        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.representation = nn.Sequential(
            nn.Conv3d(filters[0], self.mun_pro, 3, padding=1, bias=False),
            nn.BatchNorm3d(self.mun_pro),
            nn.ReLU(),
            nn.Conv3d(self.mun_pro, self.mun_pro, 1)
        )

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, inputs, mm: bool = False):
        # --- Encoder ---
        conv1 = self.conv1(inputs)           
        maxpool1 = self.maxpool1(conv1)
        
        conv2 = self.conv2(maxpool1)         
        maxpool2 = self.maxpool2(conv2)
        
        conv3 = self.conv3(maxpool2)         
        maxpool3 = self.maxpool3(conv3)
        
        conv4 = self.conv4(maxpool3)         
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)       
        center = self.dropout1(center)

        up4 = self.up_concat4(conv4, center) 
        
        # Interaction 1
        up3 = self.up_concat3(conv3, up4)
        B = inputs.shape[0]
        current_query = self.query_feat.expand(B, -1, -1) 
        current_query, pred_result_1 = self.kmax_layer1(pixel_feature=up3, query_feature=current_query)

        # Interaction 2
        up2 = self.up_concat2(conv2, up3)
        current_query, pred_result_2 = self.kmax_layer2(pixel_feature=up2, query_feature=current_query)

        # Final Decoder
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)
        
        final = self.final(up1) # (B, Num_Classes, D, H, W)
        feature_rep = self.representation(up1)
        

        prediction_result = pred_result_2
        mask_logits = prediction_result['mask_logits'] 
        class_logits = prediction_result['class_logits']
        

        prob_class = F.softmax(class_logits, dim=-1)[..., :self.n_classes] # (B, Q, C)
        

        if mask_logits.shape[-3:] != final.shape[-3:]:
            mask_logits = F.interpolate(
                mask_logits, 
                size=final.shape[-3:], 
                mode='trilinear', 
                align_corners=False
            ) # (B, Q, D, H, W)
            
        # sum_over_Q ( P(C|Q) * Mask(Q,D,H,W) ) -> (B, C, D, H, W)
        kmax_semantic_pred = torch.einsum('bqc,bqdhw->bcdhw', prob_class, mask_logits)

        if mm:
            return final, feature_rep, kmax_semantic_pred, current_query, class_logits

        return final, feature_rep