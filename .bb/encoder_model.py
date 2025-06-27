import torch
import torch.nn as nn
import torch.nn.functional
from encoder_data import PatchEmbed


class PatchProjector(torch.nn.Module):
    def __init__(self, patch_dim, embed_dim):
        super(PatchProjector, self).__init__()
        self.linear = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass through a linear layer
        to project patches to 64 dimension vectors
        essentially this is creating the embedding for the batches

        """
        x = self.linear(x)
        return x


class SelfAttentionSingle(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass of the self-attention module."""
        Q = self.q_proj(x)  # Query: B, N, D
        K = self.k_proj(x)  # Key: B, N, D
        V = self.v_proj(x)  # Value: B, N, D
        attn_scores = Q @ K.transpose(-2, -1) * (self.embed_dim**-0.5)
        attn_weights = attn_scores.softmax(dim=-1)  # (B, N, N)
        attn_weigt_drpout = self.dropout(attn_weights)
        out = attn_weigt_drpout @ V  # (B, N, D)
        return self.final_linear(out)


class SelfAttentionMulti(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # B, N, D = batch size, number of tokens, emb dim
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Applies a linear transformation to the input tensor to produce the query tensor Q using learned weights W_q
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        # Applies a linear transformation to the input tensor to produce the query tensor K using learned weights W_k
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # Applies a linear transformation to the input tensor to produce the query tensor V using learned weights W_v
        self.final_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass of the self-attention module."""
        # x: B, N, D
        B, N, D = (
            x.shape
        )  # B: batch size, N: number of tokens (patches + CLS), D: embedding dimension

        Q = self.q_proj(x)  # Query: B, N, D
        K = self.k_proj(x)  # Key: B, N, D
        V = self.v_proj(x)  # Value: B, N, D

        # Reshape Q, K, V for multi-head attention, transpose switches the dimensions
        # to optermise for multi-head computations
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, N, num_heads, head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, N, num_heads, head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, N, num_heads, head_dim)

        # compute attention scores
        attn_scores = Q @ K.transpose(-2, -1) * (self.embed_dim**-0.5)
        # (B, N, N),  @ sign is matrix multiplacation
        # self.dm ** -0.5 keeps variance
        attn_weights = attn_scores.softmax(dim=-1)  # (B, N, N)

        # apply attention weights to values
        out = attn_weights @ V  # (B, N, D)
        # apply dropout
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        # final linear
        return self.final_linear(out)


class Encoder(nn.Module):
    def __init__(self, embed_dim):  # num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttentionSingle(embed_dim)  # num_heads)
        # self.attention2 = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, tensor):
        """Forward pass of the encoder block."""
        # x shape: (B, N, D) where B is batch size, N is number of patches + 1 (CLS token), D is embedding dimension
        x_res1 = tensor
        x_res1_1 = self.norm1(x_res1)
        x_res1_2 = self.attention(x_res1_1)  # Self-attention
        tensor_2 = x_res1 + x_res1_2  # Residual connection 1

        x_res2 = tensor_2
        x_res2_1 = self.norm2(x_res2)
        x_res2_2 = self.mlp(x_res2_1)
        tensor_final = x_res2 + x_res2_2  # Residual connection 2
        return tensor_final


class Transformer(nn.Module):
    """Visual Transformer for the MNIST dataset."""

    def __init__(
        self,
        patch_dim,
        embed_dim,
        num_patches,
        # num_heads,
        num_classes=10,
        num_layers: int = "6",
    ):
        super().__init__()

        self.embed_dim = embed_dim
        # self.patch_projector = PatchProjector(patch_dim, embed_dim)
        self.patch_embed = PatchEmbed(
            patch_size=7,
            embed_dim=64,
            img_size=28,
            in_chans=1,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.ModuleList([Encoder(embed_dim) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.final_linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass of the Visual Transformer."""

        x = self.patch_embed(x)
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1 , 17
        x = torch.cat((cls_tokens, x), dim=1)  # B, 1 , 64
        x = x + self.pos_embed  # (B, 17, 64)

        for layer in self.encoder:
            x = layer(x)
        x = self.layer_norm(x)
        return self.final_linear(x[:, 0, :])  # Use CLS token for classification

        # run a loop, 6 times for 6 encoder block
