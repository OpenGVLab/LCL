from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, _expand_token
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower
from .attentive import AttentivePoolingProjection


@dataclass
class PoolProjectCfg:
    pool_proj_type: str = 'attn'
    input_dim: str = 768
    output_dim: str = 512
    attn_num_heads: int = 8


def _build_pool_project_layer(pool_project_cfg, cast_dtype: Optional[torch.dtype] = None):
    pool_project_cfg = PoolProjectCfg(**pool_project_cfg) if isinstance(pool_project_cfg, dict) else pool_project_cfg
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm

    if pool_project_cfg.pool_proj_type == 'attn':
        pool_project = AttentivePoolingProjection(
            input_dim=pool_project_cfg.input_dim,
            output_dim=pool_project_cfg.output_dim,
            num_query=1,
            num_heads=pool_project_cfg.attn_num_heads,
            norm_layer=norm_layer,
        )
    else:
        raise NotImplementedError

    return pool_project


def _build_lm_head(text_output_dim, vocab_size, cast_dtype: Optional[torch.dtype] = None):
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    lm_head = nn.Sequential(*[
        norm_layer(text_output_dim),
        nn.Linear(text_output_dim, vocab_size, bias=False),
    ])
    return lm_head


class LCL(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            pool_project_cfg: PoolProjectCfg,
            special_token_ids = {},
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        # NOTE: hack here, do not use original vision proj
        self.visual.proj = None
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.pool_project = _build_pool_project_layer(pool_project_cfg, cast_dtype)
        text_width = self.text.width
        vocab_size = text_cfg['vocab_size'] if isinstance(text_cfg, dict) else text_cfg.vocab_size
        self.lm_head = _build_lm_head(text_width, vocab_size, cast_dtype)
        # self.transformer = self.text.transformer
        # self.context_length = text.context_length
        # self.vocab_size = text.vocab_size
        # self.token_embedding = text.token_embedding
        # self.positional_embedding = text.positional_embedding
        # self.ln_final = text.ln_final
        # self.text_projection = text.text_projection
        # self.text_pool_type = text.pool_type
        # self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # special tokens ids
        self.pad_id = 0 # TODO: hack here, 0 is pad_id
        self.sot_id = special_token_ids['<start_of_text>']
        self.eot_id = special_token_ids['<end_of_text>']
        self.img_id = special_token_ids['<img_placehold>']
        self.soi_id = special_token_ids['<start_of_img>']
        self.eoi_id = special_token_ids['<end_of_img>']

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def _encode_image(self, x: torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x)
        x = x[:, 1:] # NOTE: only take patch tokens
        pooled, tokens = self.pool_project(x) # attn pool & project
        
        return pooled, tokens
    
    def _forward_text(self, x: torch.Tensor):
        cast_dtype = self.text.transformer.get_cast_dtype()
        x = x + self.text.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text.transformer(x, attn_mask=self.text.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent
    
    def get_interleaved_embs(self, image_embs, text_ids):
        cast_dtype = self.text.transformer.get_cast_dtype()

        # image token mask
        img_mask = (text_ids == self.img_id)
        # text token embs
        text_embs = self.text.token_embedding(text_ids[~img_mask]).to(cast_dtype)
        # merge to interleaved embs
        input_embs = text_embs.new_zeros((*text_ids.shape, text_embs.shape[-1]))
        input_embs[~img_mask] = text_embs
        input_embs[img_mask] = image_embs.flatten(0, 1).to(cast_dtype)

        return input_embs

    def encode_text(self, text, normalize: bool = True):
        eot_mask = (text == self.eot_id)
        # NOTE: replace <eot> with <soi> since we use text features at <soi> for contrastive learning
        text[eot_mask] = self.soi_id

        cast_dtype = self.text.transformer.get_cast_dtype()
        x = self.text.token_embedding(text).to(cast_dtype)
        x = self._forward_text(x)

        x = x[text == self.soi_id]
        x = self.text.ln_final(x)
        if self.text.text_projection is not None:
            if isinstance(self.text.text_projection, nn.Linear):
                x = self.text.text_projection(x)
            else:
                x = x @ self.text.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_contrastive_features(self, image_features, text_outputs, text_ids):
        ind_matrix = torch.arange(text_ids.shape[1], device=text_ids.device)[None].repeat((text_ids.shape[0], 1))
        # NOTE: do not take <soi> at the beginning
        soi_mask = (text_ids == self.soi_id) & (ind_matrix > 0)
        soi_batch_id, soi_seq_id = torch.nonzero(soi_mask, as_tuple=True)
        text_features = text_outputs[soi_batch_id, soi_seq_id]
        text_features = self.text.ln_final(text_features)
        if self.text.text_projection is not None:
            if isinstance(self.text.text_projection, nn.Linear):
                text_features = self.text.text_projection(text_features)
            else:
                text_features = text_features @ self.text.text_projection
        text_features = F.normalize(text_features.float(), dim=-1).to(dtype=text_features.dtype)

        # NOTE: do not take <soi> at the beginning
        soi_mask = (text_ids == self.soi_id)
        ignore_mask = (ind_matrix[soi_mask] == 0)
        image_features = image_features[~ignore_mask]
        image_features = F.normalize(image_features.float(), dim=-1).to(dtype=image_features.dtype)

        assert len(text_features) == len(image_features)
        return image_features, text_features
    
    def get_generation_logits_labels(self, text_outputs, text_ids):
        gen_mask = \
            (text_ids[:, 1:] != self.img_id) & \
            (text_ids[:, 1:] != self.eoi_id) & \
            (text_ids[:, 1:] != self.pad_id)
        
        logits = self.lm_head(text_outputs[:, :-1][gen_mask])
        labels = text_ids[:, 1:][gen_mask]

        return logits, labels

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_latent, image_embs = self._encode_image(image)
        if text is None:
            image_latent = F.normalize(image_latent, dim=-1)
            return {"image_features": image_latent, "image_embs": image_embs}
        
        interleaved_embs = self.get_interleaved_embs(image_embs, text)
        text_outputs = self._forward_text(interleaved_embs)

        image_features, text_features = self.get_contrastive_features(image_latent, text_outputs, text)
        logits, labels = self.get_generation_logits_labels(text_outputs, text)

        out_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.logit_scale.exp()
        }
        return out_dict