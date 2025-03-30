import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, d_model: int, img_size: int, patch_size: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.conv = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = int(seq_len)
        self.cls = nn.Parameter(torch.rand(1, 1, self.d_model))
        pe = torch.zeros((self.seq_len, self.d_model))
        positions = torch.arange(0, self.seq_len, dtype=torch.float32).unsqueeze(1)
        denominator = torch.pow(
            input=torch.full(size=(int(self.d_model / 2),), fill_value=10000.0),
            exponent=(torch.arange(0, self.d_model, 2) / self.d_model),
        )
        pe[:, 0::2] = torch.sin(positions / denominator)
        pe[:, 1::2] = torch.cos(positions / denominator)
        self.pe = pe.unsqueeze(0)
        self.register_buffer("pos_enc", self.pe)

    def forward(self, x):
        tok_batch = self.cls.expand(x.shape[0], -1, -1)
        toks = torch.cat([tok_batch, x], dim=1)
        x = self.pe + toks
        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.W_q = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.d_model, self.head_dim, bias=False)
        self.W_v = nn.Linear(self.d_model, self.head_dim, bias=False)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        attention = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_scores = torch.softmax(attention, dim=-1)
        attention_scores = attention_scores @ v
        return attention_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = self.d_model // self.heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.head_dim) for _ in range(self.heads)]
        )
        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.W_o(out)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.ln1 = nn.LayerNorm(self.d_model)
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.d_model),
        )

    def forward(self, x):
        out = x + self.ln1(self.mha(x))
        out = out + self.ln2(self.ffn(out))
        return out


class VisionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        img_size: int,
        patch_size: int,
        num_channels: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = (self.img_size[0] * self.img_size[1]) / (
            self.patch_size[0] * self.patch_size[1]
        )
        self.seq_len = self.num_patches + 1
        self.patch_embed = PatchEmbedding(
            self.d_model, self.img_size, self.patch_size, self.num_channels
        )
        self.pos_enc = PositionalEncoding(self.d_model, self.seq_len)
        self.transformer_encoder = nn.Sequential(
            *[
                Encoder(self.d_model, self.num_heads, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.num_classes), nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x


vit = VisionTransformer(768, 12, 1024, 8, 10, (32, 32), (16, 16), 3)
x = torch.rand((100, 3, 32, 32))
print(vit(x).shape)
