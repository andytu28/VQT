import math
from functools import partial, reduce
from operator import mul
import torch
import torch.nn as nn
import torch.nn.functional as F

from .timm_h2t_vit import TimmHead2ToeVisionTransformer


class Head2ToeVisionTransformerMAE(TimmHead2ToeVisionTransformer):

    def __init__(self, h2t_cfg, global_pool=True, **kwargs):  # Note that VPT use global-pool version of MAE pre-trained vit
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # Remove the original norm

        self.h2t_cfg = h2t_cfg
        self.num_query_tokens = h2t_cfg.NUM_QUERY_TOKENS
        self.prompt_dropout = nn.Dropout(h2t_cfg.DROPOUT)
        self.norm_feats = h2t_cfg.NORMALIZE_FEATS

        # Initiate query prompts
        if self.num_query_tokens > 0:
            patch_size = self.patch_embed.patch_size
            self.query_prompt_embeddings = nn.Parameter(torch.zeros(
                len(self.blocks), self.num_query_tokens, self.embed_dim))

            prompt_dim = self.embed_dim
            val = math.sqrt(6./float(3*reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            # xavier_uniform initialization
            nn.init.uniform_(self.query_prompt_embeddings.data, -val, val)
        else:
            self.register_parameter('query_prompt_embeddings', None)

    def train(self, mode=True):
        if mode:
            for module in self.children():
                module.eval()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)
            self.prompt_dropout.eval()

    def embeddings(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward_features(self, x):
        x = self.embeddings(x)

        B = x.shape[0]
        query_outputs = []
        for layer_idx, layer_block in enumerate(self.blocks):

            if self.query_prompt_embeddings is not None:
                q_states = self.prompt_dropout(
                        self.query_prompt_embeddings[layer_idx].expand(
                            B, -1, -1))
                x = torch.cat([q_states, x], dim=1)

            x = layer_block(x, query_prompt_len=self.num_query_tokens)

            if self.query_prompt_embeddings is not None:
                query_outputs.append(x[:, :self.num_query_tokens, :])
                x = x[:, self.num_query_tokens:, :]

        # MAE by default use global pool instead of CLS
        x = x[:, 1:, :].mean(dim=1)
        x = self.fc_norm(x)
        return x, query_outputs

    def forward(self, x, feat_select_ids=None):
        x, query_outputs = self.forward_features(x)
        B = x.shape[0]
        if self.head_dist is not None:
            raise NotImplementedError()
        else:
            included_features = [x] + [q.view(B, -1) for q in query_outputs]
            if self.norm_feats:
                included_features = [F.normalize(x) for x in included_features]
            feats = torch.cat(included_features, dim=1)

            if feat_select_ids is None:
                logits = self.head(feats)
            else:
                logits = self.head(feats[:, feat_select_ids])

            return logits
