# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision

from lib.models.stark.transformer import _get_clones, _get_activation_fn
from lib.models.stark.head import MLP
from lib.models.stark.position_encoding import gen_sineembed_for_position
from lib.models.stark.attention import MultiheadAttention
from lib.utils.box_ops import box_cxcywh_to_xyxy


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.box_embed = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                grid=None):
        output = tgt

        for layer_id, layer in enumerate(self.layers):

            reference_boxes_before_sigmoid = query_pos  # [num_queries, batch_size, 4](cxcywh)
            reference_boxes = reference_boxes_before_sigmoid.transpose(0, 1)

            obj_center = reference_boxes[..., :2].transpose(0, 1)      # [num_queries, batch_size, 2] bb中心点的坐标

            # get sine embedding for the query vector
            query_ref_boxes_sine_embed = gen_sineembed_for_position(obj_center)  # 得到每个bb中心点对应的位置编码

            memory_ = memory
            memory_h_ = memory_h
            memory_w_ = memory_w
            memory_key_padding_mask_ = memory_key_padding_mask
            pos_ = pos  # encoded特征位置编码
            grid_ = grid

            output = layer(output,
                           memory_,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           memory_key_padding_mask=memory_key_padding_mask_,
                           pos=pos_,
                           query_ref_boxes_sine_embed=query_ref_boxes_sine_embed,
                           reference_boxes=reference_boxes,
                           memory_h=memory_h_,
                           memory_w=memory_w_,
                           grid=grid_,)


        return output



class TransformerDecoderLayer(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.d_model = 256
        self.nheads = 8
        self.num_queries = 1
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.activation = _get_activation_fn(activation)

        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_qpos_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_kcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_kpos_proj = nn.Linear(self.d_model, self.d_model)
        self.sa_v_proj = nn.Linear(self.d_model, self.d_model)
        self.self_attn = MultiheadAttention(self.d_model, self.nheads, dropout=self.dropout, vdim=self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.ca_kcontent_proj = nn.Linear(self.d_model, self.d_model)
        self.ca_v_proj = nn.Linear(self.d_model, self.d_model)
        self.ca_qpos_sine_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.ca_kpos_sine_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
        self.cross_attn = MultiheadAttention(self.nheads * self.d_model, self.nheads, dropout=self.dropout, vdim=self.d_model)
        self.dropout2 = nn.Dropout(self.dropout)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.point1 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.point2 = nn.Sequential(
            nn.Linear(self.d_model // 4 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.nheads * 2),
        )
        nn.init.constant_(self.point2[-1].weight.data, 0)
        nn.init.constant_(self.point2[-1].bias.data, 0)


        self.attn1 = nn.Linear(self.d_model, self.d_model * self.nheads)
        self.attn2 = nn.Linear(self.d_model, self.d_model * self.nheads)

        # FFN
        self.linear1 = nn.Linear(self.d_model, self.dim_feedforward)
        self.dropout88 = nn.Dropout(self.dropout)
        self.linear2 = nn.Linear(self.dim_feedforward, self.d_model)
        self.dropout3 = nn.Dropout(self.dropout)
        self.norm3 = nn.LayerNorm(self.d_model)
        self.MLP = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_valid_ratio(self, mask):
        b, H, W = mask.shape
        height = torch.zeros(b)
        width = torch.zeros(b)
        mask_new = ~mask
        for idx in range(mask_new.size(0)):
            count_row = 0
            count_col = 0
            m = mask_new[idx, :]
            for i in range(m.size(1)):
                if torch.sum(m[:, i]) != 0:
                    count_col = count_col + 1
            for j in range(m.size(0)):
                if torch.sum(m[j, :]) != 0:
                    count_row = count_row + 1
            height[idx] = count_row
            width[idx] = count_col

        valid_ratio_h = height / 20
        valid_ratio_w = width / 20
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_ref_boxes_sine_embed = None,
                reference_boxes: Optional[Tensor] = None,
                memory_h=None,
                memory_w=None,
                grid=None):

        num_queries = tgt.shape[0]
        bs = tgt.shape[1]
        c = tgt.shape[2]
        n_model = c
        valid_ratio = self.get_valid_ratio(memory_key_padding_mask.view(bs, memory_h, memory_w)).cuda()

        memory_2d = memory.view(memory_h, memory_w, bs, c)
        memory_2d = memory_2d.permute(2, 3, 0, 1)  # (b,c,h,w)


        reference_boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
        reference_boxes_xyxy[:, :, 0] *= memory_w
        reference_boxes_xyxy[:, :, 1] *= memory_h
        reference_boxes_xyxy[:, :, 2] *= memory_w
        reference_boxes_xyxy[:, :, 3] *= memory_h
        reference_boxes_xyxy = reference_boxes_xyxy * valid_ratio.view(bs, 1, 4)

        q_content = torchvision.ops.roi_align(  # 根据reference_boxes_xyxy把特征图上的相应区域提取出来(7,7,256)
            memory_2d,
            list(torch.unbind(reference_boxes_xyxy, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)

        q_content_points = torchvision.ops.roi_align(
            memory_2d,
            list(torch.unbind(reference_boxes_xyxy, dim=0)),
            output_size=(7, 7),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 7, 7)

        q_content_index = q_content_points.view(bs * num_queries, -1, 7, 7)

        points = self.point1(q_content_index)
        points = points.reshape(bs * num_queries, -1)  # 把(600,64,7,7)->(600,64*7*7=3136)把小块信息汇聚成一个点
        points = self.point2(points)  # (600,3136)->(600,16)
        points = points.view(bs * num_queries, 1, self.nheads, 2).tanh()  # (600,1,8,2)

        q_content = F.grid_sample(q_content, points, padding_mode="zeros", align_corners=False).view(bs * num_queries, -1) # 根据points提供的坐标信息把q_content上的相应特征提取出来
        q_content = q_content.view(bs, num_queries, -1)   # (num_query, bs, n_head, 256)(300,2,8,256)得到8个关键点的信息
        output = self.MLP(q_content)
        return output
