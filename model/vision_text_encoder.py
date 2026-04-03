import torch
from torch import nn
import math
from copy import copy
from typing import Callable, List, Optional, Tuple, Union
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend

from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model.vitdet import get_abs_pos
from sam3.model.text_encoder_ve import text_global_pool

from .cma import CMA

class VisionTextEncoder(SAM3VLBackbone):
    def __init__(
            self,
            visual: Sam3DualViTDetNeck,
            text: VETextEncoder,
            scalp=0,
    ):
        super().__init__(
            visual=visual,
            text=text,
            compile_visual=False,
            act_ckpt_whole_vision_backbone=False,
            act_ckpt_whole_language_backbone=False,
            scalp=scalp,
        )

        self.vision_backbone = visual
        self.language_backbone = text
        self.scalp = scalp

        vis_dim = self.vision_backbone.trunk.embed_dim
        txt_dim = self.language_backbone.encoder.width

        self.fusion_stages = [0, 1, 2, 3]
        self.cma_adapters = nn.ModuleList()
        for i, stage in enumerate(self.fusion_stages):
            self.cma_adapters.append(
                CMA(
                    in_channels_vis=vis_dim,
                    in_channels_txt=txt_dim,
                    adapter_channels=256,
                )
            )


    def forward(
            self,
            samples: torch.Tensor,
            captions: List[str],
            input_boxes: Optional[torch.Tensor] = None,
            additional_text: Optional[List[str]] = None,
    ):
        num_frames = samples.shape[0]
        
        vision_feat = self._prepare_img(samples)
        tokenized, text_attention_mask, inputs_embeds, text_feat, attn_mask, text_length = self._prepare_text(captions)

        fusion_vis = [0, 8, 16, 24, 32]
        fusion_txt = [0, 6, 12, 18, 24]

        for i in range(len(fusion_vis) - 1):
            i_v_start, i_v_end = fusion_vis[i], fusion_vis[i + 1]
            i_t_start, i_t_end = fusion_txt[i], fusion_txt[i + 1]

            # Block forward
            vision_feat = self.forward_img_block(
                i_v_start, i_v_end, 
                self.vision_backbone.trunk.blocks, 
                vision_feat
            )
            text_feat = self.forward_text_block(
                i_t_start, i_t_end,
                self.language_backbone.encoder.transformer.resblocks, 
                text_feat,
                attn_mask
            )

            # CMA fusion
            if i in self.fusion_stages:
                v = vision_feat.clone().permute(0, 3, 1, 2)
                t = text_feat.clone().permute(1, 0, 2)
                
                cma_adapter = self.cma_adapters[self.fusion_stages.index(i)]
                txt_padding_mask = ~(text_attention_mask.bool())

                v, t = cma_adapter(v, t, num_frames, txt_padding_mask)
                vision_feat = vision_feat + v.permute(0, 2, 3, 1)
                text_feat = text_feat + t.permute(1, 0, 2)

        vision_output = self._post_img(vision_feat)
        text_output = self._post_text(text_feat, tokenized, text_attention_mask, inputs_embeds, text_length)

        return vision_output, text_output

    def _prepare_img(self, x: torch.Tensor):
        x = self.vision_backbone.trunk.patch_embed(x)
        h, w = x.shape[1], x.shape[2]

        if self.vision_backbone.trunk.pos_embed is not None:
            x = x + get_abs_pos(
                self.vision_backbone.trunk.pos_embed,
                self.vision_backbone.trunk.pretrain_use_cls_token,
                (h, w),
                self.vision_backbone.trunk.retain_cls_token,
                tiling=self.vision_backbone.trunk.tile_abs_pos,
            )

        x = self.vision_backbone.trunk.ln_pre(x)
        return x

    def forward_img_block(self, start, end, blk, x: torch.Tensor):
        for i in range(start, end):
            if self.vision_backbone.trunk.use_act_checkpoint and self.training:
                x = checkpoint(blk[i], x, use_reentrant=False)
            else:
                x = blk[i](x)

        return x

    def _post_img(self, x: torch.Tensor):
        x = self.vision_backbone.trunk.ln_post(x)
        s = 0
        if self.vision_backbone.trunk.retain_cls_token:
            # If cls_token is retained, we don't
            # maintain spatial shape
            x = torch.cat([self.vision_backbone.trunk.class_embedding, x.flatten(1, 2)], dim=1)
            s = 1
        feats = x[:, s:]
        if feats.ndim == 4:
            feats = feats.permute(0, 3, 1, 2)
        else:
            assert feats.ndim == 3
            h = w = math.sqrt(feats.shape[1])
            feats = feats.reshape(
                feats.shape[0], h, w, feats.shape[-1]
            ).permute(0, 3, 1, 2)

        sam3_features, sam3_pos = [], []
        sam2_features, sam2_pos = None, None
        if self.vision_backbone.sam2_convs is not None:
            sam2_features, sam2_pos = [], []
        x = feats  # simpleFPN
        for i in range(len(self.vision_backbone.convs)):
            sam3_x_out = self.vision_backbone.convs[i](x)
            sam3_pos_out = self.vision_backbone.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_features.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.vision_backbone.sam2_convs is not None:
                sam2_x_out = self.vision_backbone.sam2_convs[i](x)
                sam2_pos_out = self.vision_backbone.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_features.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)

        if self.scalp > 0:
            # Discard the lowest resolution features
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None

        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

        return output

    def _prepare_text(self, text: Union[List[str], Tuple[torch.Tensor, torch.Tensor, dict]],
                      device: torch.device = None):

        # Forward through text_encoder
        text_to_encode = copy(text)
        text_length = len(text)

        if device is None:
            device = next(self.language_backbone.parameters()).device

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            tokenized = self.language_backbone.tokenizer(
                text_to_encode,
                context_length=self.language_backbone.context_length
            ).to(device)  # [b, seq_len]

            text_attention_mask = (tokenized != 0).bool()

            # manually embed the tokens
            inputs_embeds = self.language_backbone.encoder.token_embedding(
                tokenized
            )  # [b, seq_len, d=1024]

            # TextTransformer
            seq_len = tokenized.shape[1]
            x = self.language_backbone.encoder.token_embedding(tokenized)  # [batch_size, n_ctx, d_model]

            attn_mask = self.language_backbone.encoder.attn_mask
            if attn_mask is not None:
                attn_mask = attn_mask[:seq_len, :seq_len]

            x = x + self.language_backbone.encoder.positional_embedding[:seq_len]

        return tokenized, text_attention_mask, inputs_embeds, x, attn_mask, text_length

    def _post_text(self, x: torch.Tensor, tokenized, text_attention_mask, inputs_embeds, text_length):
        output = {}
        x = self.language_backbone.encoder.ln_final(x)
        pooled, tokens = text_global_pool(x, tokenized, pool_type=self.language_backbone.encoder.pool_type)
        if self.language_backbone.encoder.text_projection is not None:
            if isinstance(self.language_backbone.encoder.text_projection, nn.Linear):
                pooled = self.language_backbone.encoder.text_projection(pooled)
            else:
                pooled = pooled @ self.language_backbone.encoder.text_projection

        text_memory = tokens
        assert text_memory.shape[1] == inputs_embeds.shape[1]
        # Invert attention mask because its the opposite in pytorch transformer
        text_attention_mask = text_attention_mask.ne(1)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = text_memory.transpose(0, 1)
        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory = self.language_backbone.resizer(text_memory)

        inputs_embeds = inputs_embeds.transpose(0, 1)
        text_embeds = inputs_embeds

        text_memory = text_memory[:, : text_length]
        text_attention_mask = text_attention_mask[: text_length]
        text_embeds = text_embeds[:, : text_length]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = (
            text_embeds  # Text embeddings before forward to the encoder
        )

        return output

    def forward_text_block(self, start, end, blk, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for i in range(start, end):
            if (
                    self.language_backbone.encoder.transformer.grad_checkpointing
                    and not torch.jit.is_scripting()
                    and self.training
            ):
                x = checkpoint(blk[i], x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk[i](x, attn_mask=attn_mask)
        return x



