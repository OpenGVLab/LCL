
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, 
            dim, 
            num_heads=8, 
            qkv_bias=False, 
            qk_scale=None, 
            attn_head_dim=None, 
            out_dim=None, 
            out_bias=True
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == out_dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.proj = nn.Linear(all_head_dim, out_dim, bias=out_bias)

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)    # (B, N_head, N_q, dim)

        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)   
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))      # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1) 
        x = self.proj(x)

        return x


class AttentiveBlock(nn.Module):

    def __init__(self, 
            dim, 
            num_heads, 
            qkv_bias=False, 
            qk_scale=None, 
            norm_layer=nn.LayerNorm,
            attn_head_dim=None, 
            out_dim=None, 
            out_bias=True
    ):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            attn_head_dim=attn_head_dim, 
            out_dim=out_dim, 
            out_bias=out_bias
        )
        
    def forward(self, x_q, x_kv, pos_q, pos_k):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        return x


class AttentivePoolingProjection(nn.Module):
    def __init__(self,
            input_dim,
            output_dim,
            num_query,
            num_heads=None,
            norm_layer=nn.LayerNorm,
            out_bias=False,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = int(output_dim // 64)
        self.query_token = nn.Parameter(torch.randn(1, num_query, input_dim))
        self.pooler = AttentiveBlock(
                dim=input_dim, 
                out_dim=output_dim,
                num_heads=num_heads, 
                qkv_bias=True, 
                qk_scale=None,
                norm_layer=norm_layer,
                out_bias=out_bias,
            )
        
    def forward_pool(self, x):
        query_tokens = self.query_token.expand(x.shape[0], -1, -1)
        query_tokens = self.pooler(query_tokens, x, 0, 0)
        return query_tokens.squeeze(1)
        
    def forward_project(self, x):
        x = self.pooler.norm1_v(x)
        x = F.linear(input=x, weight=self.pooler.cross_attn.v.weight, bias=self.pooler.cross_attn.v_bias)
        x = self.pooler.cross_attn.proj(x)
        return x
    
    def forward(self, x):
        pooled = self.forward_pool(x)
        projected = self.forward_project(x)
        return pooled, projected
