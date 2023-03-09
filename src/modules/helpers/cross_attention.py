import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x, y, mask):

        x_b, x_t, x_e = x.size()
        y_b, y_t, y_e = y.size()

        b, t, e = x.size()
        h = self.heads
        keys = self.tokeys(x).view(x_b, x_t, self.head, self.emb)
        queries = self.toqueries(y).view(y_b, y_t, self.head, self.emb)
        values = self.tovalues(x).view(x_b, x_t, self.head, self.emb)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(x_b * self.head, x_t, self.emb)
        queries = queries.transpose(1, 2).contiguous().view(y_b * self.head, y_t, self.emb)
        values = values.transpose(1, 2).contiguous().view(x_b * self.head, x_t, self.emb)

        queries = queries / (y_e ** (1 / 4))
        keys = keys / (x_e ** (1 / 4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (y_b * self.heads, y_t, x_t)

        if self.mask:  # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        if mask is not None:
            dot = dot.masked_fill(mask == 0, -1e9)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(y_b, self.heads, y_t, self.emb)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(y_b, y_t, self.heads * self.emb)

        return self.unifyheads(out)

def mask_(matrices, maskval=0.0, mask_diagonal=True):

    b, h, w = matrices.size()
    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval