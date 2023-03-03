import torch
import os
from easydict import EasyDict
from functools import partial, reduce
import numpy as np
import torch.nn as nn
from ..utils import logging
from .vit_backbones.h2t_vit import Head2ToeVisionTransformer
from .vit_backbones.h2t_vit_mae import Head2ToeVisionTransformerMAE
from .build_vit_backbone import MODEL_ZOO
from .build_model import load_model_to_device
from .mlp import MLP
from timm.models.vision_transformer import _cfg


logger = logging.get_logger("visual_prompt")


def build_h2t_mae_model(model_type, crop_size, h2t_cfg, model_root,
                        load_pretrain=True, vis=False, combine_method='concat'):
    if combine_method == 'concat':
        num_q = 1 if h2t_cfg.POOLING_FEATS or h2t_cfg.WEIGHTED_SUM_FEATS else h2t_cfg.NUM_QUERY_TOKENS
        m2featdim = {
            'mae_vitb16': int((768+768*12*num_q)*h2t_cfg.KEEP_FRAC),
        }
    else:
        assert(h2t_cfg.KEEP_FRAC == 1.0)
        m2featdim = {
            'mae_vitb16': 768,
        }

    model = Head2ToeVisionTransformerMAE(
            h2t_cfg=h2t_cfg, drop_path_rate=0.1,
            global_pool=True,  # default settings for mae-finetune
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

    ckpt = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f'Loading the checkpoint and get the message: {str(msg)}')
    model.head = nn.Identity()
    return model, m2featdim[model_type]

def build_h2t_vit_sup_models(model_type, crop_size, h2t_cfg, model_root,
                             load_pretrain=True, vis=False, combine_method='concat'):
    if combine_method == 'concat':
        num_q = 1 if h2t_cfg.POOLING_FEATS or h2t_cfg.WEIGHTED_SUM_FEATS else h2t_cfg.NUM_QUERY_TOKENS
        m2featdim = {
            'sup_vitb16_imagenet21k': int((768 + 768*12*num_q)*h2t_cfg.KEEP_FRAC),
            'sup_vitb16_imagenet1k' : int((768 + 768*12*num_q)*h2t_cfg.KEEP_FRAC),
            'sup_vitl16_imagenet21k': int((1024 + 1024*24*num_q)*h2t_cfg.KEEP_FRAC),
            'sup_vith14_imagenet21k': int((1280 + 1280*32*num_q)*h2t_cfg.KEEP_FRAC),
        }
    else:
        assert(h2t_cfg.KEEP_FRAC == 1.0)
        m2featdim = {
            'sup_vitb16_imagenet21k': 768,
            'sup_vitb16_imagenet1k' : 768,
            'sup_vitl16_imagenet21k': 1024,
            'sup_vith14_imagenet21k': 1028,
        }
    assert(h2t_cfg is not None)
    model = Head2ToeVisionTransformer(
            model_type, h2t_cfg, crop_size, num_classes=-1, vis=vis, combine_method=combine_method)

    if load_pretrain:
        model.load_from(np.load(os.path.join(
            model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]


class H2TViT(nn.Module):

    def __init__(self, cfg, load_pretrain=True, vis=False,
                 combine_method='concat'):
        super(H2TViT, self).__init__()
        self.cfg = cfg
        h2t_cfg = cfg.MODEL.H2T
        self.combine_method = combine_method
        self.build_backbone(
                h2t_cfg, cfg, load_pretrain, vis)
        self.side = None
        self.setup_head(cfg)

    def build_backbone(self, h2t_cfg, cfg, load_pretrain, vis):
        assert(cfg.MODEL.TRANSFER_TYPE == 'h2t-prompt')
        self.enc, self.feat_dim = build_h2t_vit_sup_models(
                cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, h2t_cfg, cfg.MODEL.MODEL_ROOT,
                load_pretrain, vis, combine_method=self.combine_method)

        for k, p in self.enc.named_parameters():
            if 'prompt' in k:
                p.requires_grad = True
            elif 'combine_selfatten_block' in k:
                p.requires_grad = True
            elif 'layerwise_mlps' in k:
                p.requires_grad = True
            elif 'layer_position_embeddings' in k:
                p.requires_grad = True
            elif 'combine_params' in k:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def setup_head(self, cfg):
        self.head = MLP(
                input_dim=self.feat_dim,
                mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
                        [cfg.DATA.NUMBER_CLASSES], #noqa
                special_bias=True
        )

    def forward(self, x, feat_select_ids=None):
        x = self.enc(x, feat_select_ids=feat_select_ids)
        x = self.head(x)
        return x


class H2TSSLViT(H2TViT):

    def __init__(self, cfg, combine_method='concat'):
        super(H2TSSLViT, self).__init__(
                cfg=cfg, combine_method=combine_method)

    def build_backbone(self, h2t_cfg, cfg, load_pretrain, vis):
        assert(cfg.MODEL.TRANSFER_TYPE == 'h2t-prompt')

        if 'mae' in cfg.DATA.FEATURE:
            build_fn = build_h2t_mae_model
        else:
            raise NotImplementedError()

        self.enc, self.feat_dim = build_fn(
                cfg.DATA.FEATURE, cfg.DATA.CROPSIZE, h2t_cfg, cfg.MODEL.MODEL_ROOT,
                load_pretrain, vis, combine_method=self.combine_method)

        for k, p in self.enc.named_parameters():
            if 'prompt' in k:
                p.requires_grad = True
            elif 'combine_selfatten_block' in k:
                p.requires_grad = True
            elif 'layerwise_mlps' in k:
                p.requires_grad = True
            elif 'layer_position_embeddings' in k:
                p.requires_grad = True
            elif 'combine_params' in k:
                p.requires_grad = True
            else:
                p.requires_grad = False


def build_head2toe_model(cfg, combine_method='concat'):

    if cfg.MODEL.TYPE == 'h2t-vit':
        model = H2TViT(
                cfg, combine_method=combine_method)
    elif cfg.MODEL.TYPE == 'h2t-ssl-vit':
        model = H2TSSLViT(
                cfg, combine_method=combine_method)
    else:
        raise NotImplementedError()

    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")
    return model, device
