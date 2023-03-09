import torch.nn as nn
import torch.nn.functional as F
import torch

from .self_attention import SelfAttention
from .cross_attention import CrossAttention

class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, cross_attention=False):
        super().__init__()

        self.cross = cross_attention

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        if self.cross:
            self.cross_attention = CrossAttention(emb, heads=heads, mask=mask)
            self.norm3 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x, mask, enc_out=None):

        attended = self.attention(x, mask)

        x = self.norm1(attended + x)
        x = self.do(x)

        if self.cross:
            attended = self.cross_attention(x, enc_out)
            x = self.norm3(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x, mask

class TransformerEncoder(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, mask=False))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, h, mask):

        tokens = torch.cat((x, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks((tokens, mask))

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x

class TransformerDecoder(nn.Module):

    def __init__(self, input_dim, emb, heads, depth, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(emb=emb, heads=heads, mask=False, cross_attention=True))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, output_dim)

    def forward(self, x, enc, h, mask):

        tokens = torch.cat((x, h), 1)

        b, t, e = tokens.size()

        x, mask = self.tblocks(tokens, mask, enc)

        x = self.toprobs(x.view(b * t, e)).view(b, t, self.num_tokens)

        return x, tokens