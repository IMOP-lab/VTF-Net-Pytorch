# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
# import sys
# sys.path('../Model')
import math
from .attention import SEAttention
import torch
from torchvision import models
import torch.nn as nn
from .resnet import resnet34
from torch.nn import functional as F
import torchsummary
from torch.nn import init
from .gap import GlobalAvgPool2D
up_kwargs = {'mode': 'bilinear', 'align_corners': True}
from .xLSTM import ViLBlock,SequenceTraversal,DropPath
from torch.cuda.amp import autocast

class ViLLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, channel_token = False):
        super().__init__()
        print(f"ViLLayer: dim: {dim}")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        out = x_vil.transpose(-1, -2).reshape(B, d_model, *img_dims)

        return out

    def forward_channel_token(self, x):
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_vil = self.vil(x_flat)
        out = x_vil.reshape(B, n_tokens, *img_dims)

        return out

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)

        return out


class xLSTMOperation(nn.Module):
    def __init__(self, input_channels, use_spatial=True):
        super(xLSTMOperation, self).__init__()
        self.use_spatial = use_spatial
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(input_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.linear = nn.Linear(input_channels, input_channels)
        self.silu = nn.SiLU(inplace=True)
        if self.use_spatial:
            self.spatial_layer = ViLLayer(input_channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Reshape for ViLLayer
        x1 = x.view(batch_size, channels, -1).permute(0, 2, 1)  # 变换为 [batch_size, height*width, channels]
        x1 = self.spatial_layer(x1)  # 使用ViLLayer处理空间信息
        x1 = x1.permute(0, 2, 1).view(batch_size, channels, height, width)  # 变换回原始形状

        # Linear and SiLU operations
        x2 = x.view(batch_size, -1)
        x2 = self.linear(x2)
        x2 = self.silu(x2)
        x2 = x2.view(batch_size, channels, height, width)
        
        # Combine paths
        x = x1 + x2
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
   

#对每一层都做xLSTM
class VTFE(nn.Module):
    def __init__(self, input_channels_list, n_stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage):
        super(VTFE, self).__init__()
        self.stages = nn.ModuleList()
        self.xlstm_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        for s in range(n_stages):
            stage = nn.Sequential()
            in_channels = input_channels_list[s]
            for b in range(n_blocks_per_stage[s]):
                stage.add_module(f"block{b + 1}", nn.Sequential(
                    conv_op(in_channels, features_per_stage[s], kernel_size=kernel_sizes[s], stride=strides[s], padding=1),
                    nn.BatchNorm2d(features_per_stage[s]),
                    nn.ReLU(inplace=True)
                ))
                in_channels = features_per_stage[s]

            self.stages.append(stage)
            # 对每个stage添加xLSTMOperation
            self.xlstm_layers.append(xLSTMOperation(features_per_stage[s]))
            self.linear_layers.append(nn.Linear(features_per_stage[s], features_per_stage[s]))  # 更新尺寸

    def forward(self, x):
        outputs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # 对每个stage进行xLSTMOperation
            residual = x.view(x.size(0), -1)
            x = self.linear_layers[i](residual)
            x = x.view(residual.size(0), x.size(2), x.size(3), -1)
            x = torch.flip(x, dims=[-1])
            x = self.xlstm_layers[i](x)
            x += residual.view_as(x)  # 残差连接，确保尺寸匹配
            outputs.append(x)
        return outputs
    

class VTFNet(nn.Module):
    def __init__(self, num_class=1):
        super(VTFNet, self).__init__()
        
        out_planes = num_class * 8  # num_class=2, 16
        self.backbone = resnet34(pretrained=True)
        self.msaf_module = MSAF_Module(512, 512, out_planes)
        self.rsr5 = RescaledSEResidualBlock(512, 256, relu=False, last=True)  # 256
        self.rsr4 = RescaledSEResidualBlock(256, 128, relu=False)  # 128
        self.rsr3 = RescaledSEResidualBlock(128, 64, relu=False)  # 64
        self.rsr2 = RescaledSEResidualBlock(64, 64)
        self.gap = GlobalAvgPool2D()
        self.adu1 = AttentionDepthwiseUnit(512, 512, 512)
        self.adu2 = AttentionDepthwiseUnit(512, 256, 256)
        self.adu3 = AttentionDepthwiseUnit(256, 128, 128)
        self.adu4 = AttentionDepthwiseUnit(128, 64, 64)
        self.adu5 = AttentionDepthwiseUnit(64, 64, 64)

        self.relu = nn.ReLU()
        self.final_decoder = EFRE(in_channels=[64, 64, 128, 256, 512], out_channels=32, in_feat_output_strides=(4, 8, 16, 32), out_feat_output_stride=4, norm_fn=nn.BatchNorm2d, num_groups_gn=None)
        self.final_conv = nn.Conv2d(32, out_planes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(512, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        self.direc_reencode = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1),
        )
        self.encoder = VTFE(
            input_channels_list=[64, 64, 128, 256, 512],  # 每一层的输入通道数
            n_stages=5,  # 假设我们有4个分辨率阶段
            features_per_stage=[64, 64, 128, 256, 512],  # 假设每个阶段的特征图通道数
            conv_op=nn.Conv2d,  # 使用默认的2D卷积
            kernel_sizes=[3, 3, 3, 3, 3],  # 假设每个阶段的卷积核大小为3x3
            strides=[1, 1, 1, 1, 1],  # 假设每个阶段的步长为1
            # n_blocks_per_stage=[2, 2, 2, 2, 2]  # 假设每个阶段有2个残差块
            n_blocks_per_stage=[1, 1, 1, 1, 1]  # 假设每个阶段有2个残差块
        )
        
    def forward(self, x):
        # x [2, 3, 512, 512] if batch_size=2   # 3
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)  # 1/4   64
        c3 = self.backbone.layer2(c2)  # 1/8   128
        c4 = self.backbone.layer3(c3)  # 1/16   256
        c5 = self.backbone.layer4(c4)  # 1/32   512
        

        # xLSTM Encoder
        c1_enc = self.encoder.stages[0](c1)  # VTFE 的第一个 stage
        c2_enc = self.encoder.stages[1](c2)  # VTFE 的第二个 stage
        c3_enc = self.encoder.stages[2](c3)  # VTFE 的第三个 stage
        c4_enc = self.encoder.stages[3](c4)  # VTFE 的第四个 stage
        
        # Residual Connections
        c1 = c1 + c1_enc 
        c2 = c2 + c2_enc 
        c3 = c3 + c3_enc 
        c4 = c4 + c4_enc 
       
        #### directional Prior ####
        directional_c5 = self.channel_mapping(c5)  # 16 
        mapped_c5 = F.interpolate(directional_c5, scale_factor=32, mode='bilinear', align_corners=True)  # 16
        mapped_c5 = self.direc_reencode(mapped_c5)  # 16
        
        d_prior = self.gap(mapped_c5)  # 16

        c5 = self.msaf_module(c5,d_prior)  # d_prior被reencoder，通道数变成512

        c6 = self.gap(c5)  # 改变了height和width 【2，512，1，1】
        
        r5 = self.adu1(c6, c5)  # 512

        b4 = self.adu2(self.gap(r5), c4)
        d4 = self.rsr5(c5)
        r4 = self.relu(self.gap(b4) + d4)
        
        b3 = self.adu3(self.gap(r4), c3)
        d3 = self.rsr4(c4)
        r3 = self.relu(self.gap(b3) + d3)
        
        b2 = self.adu4(self.gap(r3), c2)
        d2 = self.rsr3(c3)
        r2 = self.relu(self.gap(b2) + d2)
        
        b1 = self.adu5(self.gap(r2), c1)
        d1 = self.rsr2(c2)
        r1 = self.relu(self.gap(b1) + d1)

        feat_list = [r1, r2, r3, r4, c5]
        encoder_feats = [c1, c2, c3, c4, c5]
        final_feat = self.final_decoder(feat_list, encoder_feats) # 32

        final_out = self.final_conv(final_feat)
        final_out = self.upsample4x_op(final_out)  # out_planes = 16

        return final_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

class EFRE(nn.Module):
    def __init__(self, in_channels, out_channels, in_feat_output_strides=(4, 8, 16, 32), out_feat_output_stride=4, norm_fn=nn.BatchNorm2d, num_groups_gn=None):
        super(EFRE, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        dec_level = 0
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))
            num_layers = num_upsample if num_upsample != 0 else 1
            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    DepthwiseSeparableConv(in_channels[dec_level] if idx == 0 else out_channels, out_channels, 3, 1, 1),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                    AttentionModule(out_channels)
                ) for idx in range(num_layers)
            ]))
            dec_level += 1
    def forward(self, feat_list: list, encoder_feats: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            # 如果是最后一层特征c5，确保其上采样后尺寸一致
            if idx >= len(feat_list):
                decoder_feat = block(feat_list[-1] + encoder_feats[-1])
            else:
                decoder_feat = block(feat_list[idx] + encoder_feats[idx])
            inner_feat_list.append(decoder_feat)    
        out_feat = sum(inner_feat_list) / len(inner_feat_list)
        return out_feat
    
class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn

    
class MSAF_Module(nn.Module):  # MultiAttentionFusionModule
    def __init__(self, in_channels, out_channels, num_class):
        super(MSAF_Module, self).__init__()
        self.inter_channels = in_channels // 8

        # Example: Using Enhanced Attention Mechanisms
        self.att_layers = nn.ModuleList([TemporalFrequencyAttention(self.inter_channels, self.inter_channels) for _ in range(8)])
        # Multi-Scale Feature Fusion
        self.multi_scale_fusion = MultiScaleFeatureFusion(in_channels, self.inter_channels)
        # Final Convolution
        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(self.inter_channels, out_channels, 1))
        # Re-encoding for class number compatibility
        inter_channels = num_class * 8 if num_class < 32 else in_channels
        self.reencoder = nn.Sequential(nn.Conv2d(num_class, inter_channels, 1), nn.ReLU(True), nn.Conv2d(inter_channels, in_channels, 1))

    def forward(self, x, d_prior):
        # Re-encode class information
        enc_feat = self.reencoder(d_prior)
        # Apply attention mechanisms
        feats = [att(x[:, i*self.inter_channels:(i+1)*self.inter_channels], enc_feat[:, i*self.inter_channels:(i+1)*self.inter_channels]) for i, att in enumerate(self.att_layers)]
        # Perform multi-scale feature fusion
        fused_feat = self.multi_scale_fusion(feats)
        # Apply final convolution
        output = self.final_conv(fused_feat)
        # Add skip connection
        output += x

        return output
    
class TemporalFrequencyAttention(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(TemporalFrequencyAttention, self).__init__()
        self.time_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.freq_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.final_conv = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), norm_layer(inter_channels), nn.ReLU())
        self.outconv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))
        self.fft_attention = FFTAttentionModel(inter_channels)

    def forward(self, x, enc_feat):
        fft_feat = self.fft_attention(x)
        fft_conv = self.final_conv(fft_feat)
        combined_feat =  fft_conv 
        combined_feat = combined_feat * F.sigmoid(enc_feat)
        output = self.outconv(combined_feat)
        
        return output
    

class FFTAttentionModel(nn.Module):
    def __init__(self, channel):
        super(FFTAttentionModel, self).__init__()
        # 假设输入是实数和虚数部分堆叠而成，因此通道数翻倍
        self.lsk_block = LSKblock(channel * 2)  # 因为 FFT 后实部和虚部堆叠
    def forward(self, x):
        # FFT变换，假设x是实数
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 将复数频域数据转换为实数形式（实部和虚部堆叠）
        x_fft_real_imag = torch.cat((x_fft_shifted.real, x_fft_shifted.imag), dim=1)

        # 应用SKAttention
        # x_fft_attended = self.sk_attention(x_fft_real_imag)
        x_fft_attended = self.lsk_block(x_fft_real_imag)

        # 可能需要将处理后的数据转换回复数形式进行IFFT
        # 这里仅展示了结构，实际应用时需要根据情况调整
        real_part = x_fft_attended[:, :x_fft_attended.size(1) // 2, :, :]
        imag_part = x_fft_attended[:, x_fft_attended.size(1) // 2:, :, :]
        x_fft_attended_complex = torch.complex(real_part, imag_part)

        # IFFT变换
        x_ifft_shifted = torch.fft.ifftshift(x_fft_attended_complex, dim=(-2, -1))
        y_ifft = torch.fft.ifftn(x_ifft_shifted, dim=(-2, -1))
        y = y_ifft.real  # 取实部

        return y

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()
        self.conv = nn.Conv2d(in_channels , out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        # 假设features是一个由8个特征组成的列表
        x = torch.cat(features, dim=1)  # 沿着通道维度拼接
        x = self.conv(x)  # 将通道数减少到所需的输出通道数
        return x


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_mask = self.attention(x)
        return x * attention_mask

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AttentionDepthwiseUnit(nn.Module):
    def __init__(self, in_channels, channel_in, out_channels, AdaptiveProjection=False):
        super(AttentionDepthwiseUnit, self).__init__()
        self.AdaptiveProjection = AdaptiveProjection
        self.vista_processor  = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=1), nn.ReLU(True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=1)
        )
        self.element_transformer  = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1), nn.BatchNorm2d(out_channels),
            nn.ReLU(True), nn.Dropout(0.5)
        )
        self.feature_enhancer   = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1), nn.BatchNorm2d(out_channels),
            nn.ReLU(True), nn.Dropout(0.5)
        )
        self.normalizer = nn.Sigmoid()
        self.attention_mask = AttentionModule(out_channels)
        
    def forward(self, vista_feature, elements):
        identity = elements
        element_transformed  = self.element_transformer(elements)
        vista_transformed  = self.vista_processor (vista_feature)
        vista_transformed  = self.attention_mask(vista_transformed)
        attention_weights  = self.normalizer((vista_transformed * element_transformed).sum(dim=1, keepdim=True))
        enhanced_features   = self.feature_enhancer(elements)
        output  = attention_weights  * enhanced_features 
        if output .shape == identity.shape:
            output  += identity
        return output 

    
class RescaledSEResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d, scale=2, relu=True, last=False):
        super(RescaledSEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True) if relu else nn.Identity()
        self.se = SEAttention(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True) if relu else nn.Identity()
        self.scale = scale
        self.last = last
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        identity = x
        if not self.last:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.se(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        if x.shape == identity.shape:
            x += identity
        return x    
    
