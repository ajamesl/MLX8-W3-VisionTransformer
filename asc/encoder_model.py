import torch
import torch.nn as nn
import torch.nn.functional
from config import get_hyperparams
from encoder_data import PatchEmbed

hp = get_hyperparams()

batch_size = hp["batch_size"]
epochs = hp["epochs"]
lr = hp["learning_rate"]
patch_size = hp["patch_size"]
embed_dim = hp["embed_dim"]
num_heads = hp["num_heads"]
num_layers = hp["num_layers"]
num_classes = hp["num_classes"]
img_size = hp["img_size"]
data_path = hp["data_path"]
num_patches = hp["num_patches"]


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

    def forward(self, q_input, k_input, v_input, attn_mask=None):
        """
        q_input, k_input, v_input: tensors of shape (B, N, D)
        attn_mask: optional boolean mask of shape (B, num_heads, N, N) or (N, N)
        """
        # x: B, N, D
        B, N, D = q_input.shape
        #print(f"q_input shape: {q_input.shape}")
        #print(f"k_input shape: {k_input.shape}")
        #print(f"v_input shape: {v_input.shape}")

        B, N_q, D = q_input.shape
        _, N_k, _ = k_input.shape
        _, N_v, _ = v_input.shape

        Q = self.q_proj(q_input).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(k_input).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(v_input).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)

        # compute attention scores
        attn_scores = Q @ K.transpose(-2, -1) * (self.embed_dim**-0.5)
        # Apply mask if provided: mask should contain True where positions are masked
        if attn_mask is not None:
            # Convert mask to float -inf where mask is True to prevent attention
            attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

        attn_weights = attn_scores.softmax(dim=-1)  # (B, N, N)

        # apply attention weights to values
        out = attn_weights @ V  # (B, N, D)
        # apply dropout
        out = self.dropout(out)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        # final linear
        return self.final_linear(out)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):  # num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttentionMulti(embed_dim, num_heads)
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
        x_res1_2 = self.attention(x_res1_1, x_res1_1, x_res1_1)  # Self-attention
        tensor_2 = x_res1 + x_res1_2  # Residual connection 1

        x_res2 = tensor_2
        x_res2_1 = self.norm2(x_res2)
        x_res2_2 = self.mlp(x_res2_1)
        tensor_final = x_res2 + x_res2_2  # Residual connection 2
        return tensor_final


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.masked_attn = SelfAttentionMulti(
            embed_dim,
            num_heads,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = SelfAttentionMulti(
            embed_dim,
            num_heads,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x, enc_out, mask):
        x_res1 = x
        x = self.ln1(x)
        x = self.masked_attn(x, x, x, attn_mask=mask)
        x = x + x_res1

        x_res2 = x
        x = self.cross_attn(x, enc_out, enc_out)
        x = self.ln2(x) + x_res2

        x_res3 = x
        x = self.mlp(x)
        x = x + x_res3
        return x


class Transformer(nn.Module):
    """Visual Transformer for the MNIST dataset."""

    def __init__(
        self,
        patch_size,
        seq_len,
        embed_dim,
        num_heads,
        num_layers,
        num_classes,
        img_size,
        in_chans=1,
        
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, embed_dim, img_size, in_chans)
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.encoder = nn.ModuleList(
            [Encoder(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.final_linear = nn.Linear(embed_dim, num_classes)

        self.seq_len = seq_len
        self.vocab_size = num_classes + 1  # +1 for start token
        self.tok_embed = nn.Embedding(self.vocab_size, embed_dim)

        self.pos_encod_dec = nn.Parameter(torch.zeros(1, seq_len, embed_dim))  # WHY?
        nn.init.trunc_normal_(self.pos_encod_dec, std=0.02)

        self.decoder = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.linear = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, x, y):
        # x: (B, 1, 256, 256)
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, 256, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 257, embed_dim)
        x = x + self.pos_embed  # (B, 257, embed_dim)
        for block in self.encoder:
            x = block(x)
        x = self.layer_norm(x)  # (B, 257, embed_dim)

        # Create start token (last token in vocabulary)
        start_token = torch.full((B, 1), self.vocab_size-1, device=y.device, dtype=torch.long)

        # Shift targets right: [start] + targets[:-1]
        decoder_input = torch.cat([start_token, y[:, :-1]], dim=1)  # (B, seq_len)

        # Embed shifted inputs
        y_embed = self.tok_embed(decoder_input)  # (B, seq_len, embed_dim)
        y_embed = y_embed + self.pos_encod_dec.expand(B, -1, -1)  # Add positional encoding
        
        mask = torch.triu(
            torch.ones((self.seq_len, self.seq_len), device=x.device), diagonal=1
        ).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, seq_len, seq_len)
        mask = mask.expand(
            B, self.encoder[0].attention.num_heads, self.seq_len, self.seq_len
        )

        for block in self.decoder:
            y_embed = block(y_embed, x, mask=mask) # Process shifted inputs
        out = self.linear(y_embed)  # (B, seq_len, vocab_size)
        return out
