# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn


from lib.models.stark.sam_decoder import TransformerDecoder, TransformerDecoderLayer


class Transformer(nn.Module):
    def __init__(self, cfg, activation="relu"):
        super().__init__()
        self.d_model = cfg.MODEL.HIDDEN_DIM
        self.nheads = 8
        self.num_queries = 1
        self.dec_layers = 1
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = 0.1


        decoder_layer = TransformerDecoderLayer(activation)
        self.decoder = TransformerDecoder(decoder_layer, self.dec_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, query_embed, pos_embeds):
        a = self.forward_single_scale(srcs, masks, query_embed, pos_embeds)
        return a

    def forward_single_scale(self, src, mask, query_embed, pos):
        bs, c, memory_h, memory_w = src.shape


        grid = None

        src = src.flatten(2).permute(2, 0, 1)  # flatten NxCxHxW to HWxNxC

        tgt = torch.zeros(self.num_queries, bs, c, device=query_embed.device)

        memory = src

        # decoder
        output = self.decoder(tgt,
                              memory,
                              memory_key_padding_mask=mask,
                              pos=pos,
                              query_pos=query_embed,
                              memory_h=memory_h,
                              memory_w=memory_w,
                              grid=grid)
        return output.unsqueeze(0)



def build_transformer(args):
    return Transformer(args, activation="relu")
