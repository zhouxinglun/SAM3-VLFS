import torch
import einops
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional

class AdapterCA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                kv_pos: Optional[Tensor] = None
            ):
        att = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, kv_pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * att
        return tgt


class CMA(nn.Module):
    def __init__(
            self,
            in_channels_vis: int,
            in_channels_txt: int,
            adapter_channels: int,
    ):
        super().__init__()

        # vision project
        self.proj_vis_down = nn.Sequential(
            nn.Conv2d(in_channels_vis, adapter_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(adapter_channels),
            nn.ReLU(True)
        )
        self.proj_vis_up = nn.Sequential(
            nn.Conv2d(adapter_channels, in_channels_vis, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels_vis),
        )

        # text project
        self.proj_txt_down = nn.Linear(in_channels_txt, adapter_channels, bias=False)
        self.proj_txt_up = nn.Linear(adapter_channels, in_channels_txt, bias=False)

        # cross modal attention
        self.ca_V2T = AdapterCA(d_model=adapter_channels, nhead=8, dropout=0.0)
        self.ca_T2V = AdapterCA(d_model=adapter_channels, nhead=8, dropout=0.0)

        # Channel-wise gating
        self.gate_vis = nn.Parameter(torch.zeros(adapter_channels))
        self.gate_txt = nn.Parameter(torch.zeros(adapter_channels))

        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            vis: torch.Tensor,
            txt: torch.Tensor,
            shot: int,
            txt_padding_mask: Optional[torch.Tensor] = None
    ):
        BT, C, H, W = vis.size()
        B = BT // shot

        x_vis = self.proj_vis_down(vis)     # [B*T, C_adp, H, W]
        x_txt = self.proj_txt_down(txt)     # [L, B, C_adp]

        # Vision to Text
        q_vis = einops.rearrange(x_vis, 'bt c h w -> (h w) bt c')
        kv_txt = x_txt.repeat_interleave(shot, dim=1)
        txt_padding_mask = txt_padding_mask.repeat_interleave(shot, dim=0)
    
        vis_att_out = self.ca_V2T(tgt=q_vis, memory=kv_txt, memory_key_padding_mask=txt_padding_mask)
        vis_att_out = einops.rearrange(vis_att_out, '(h w) bt c -> bt c h w', h=H, w=W)

        # Text to Vision
        vis_set_features = einops.rearrange(x_vis, '(b t) c h w -> (h w) b t c', t=shot).mean(dim=2)   # [HW, B, C_adp]
        txt_att_out = self.ca_T2V(x_txt, memory=vis_set_features)

        # Gated Residual
        gate_vis = self.sigmoid(self.gate_vis).view(1, -1, 1, 1)
        gate_txt = self.sigmoid(self.gate_txt).view(1, 1, -1)

        vis_out = self.proj_vis_up(vis_att_out * gate_vis)
        txt_out = self.proj_txt_up(txt_att_out * gate_txt)

        return vis_out, txt_out