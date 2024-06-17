from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def sign_to_bin(x):
    return 0.5 * (1 - x)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, x_e, mask):
        for idx, layer in enumerate(self.layers, start=1):
            x, x_e = layer(x, x_e, mask)
            if idx == len(self.layers)//2 and len(self.layers) > 1:
                x = self.norm2(x)
        return self.norm(x), x_e


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # Residual NN for MH-Sa and Feed Fwd output place


class EncoderLayer(nn.Module):  # attention iteration
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.norm = LayerNorm(size)

    def forward(self, x, e_x, mask):  # x: left side, e_x: right side
        x = self.sublayer[0](x, lambda x: self.self_attn(self.norm(x), e_x, e_x, mask.transpose(-2, -1)))
        x = self.sublayer[1](x, self.feed_forward)

        e_x = self.sublayer[0](e_x, lambda e_x: self.self_attn(self.norm(e_x), x, x, mask))
        e_x = self.sublayer[1](e_x, self.feed_forward)
        return x, e_x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # embedding dimension must be divisible by number of heads
        self.d_k = d_model // h
        self.h = h  # number of heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()

        code = args.code
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model*4, dropout)

        self.src_embed_m = torch.nn.Parameter(torch.empty((code.n, args.d_model)))
        self.src_embed_s = torch.nn.Parameter(torch.empty((code.pc_matrix.size(0), args.d_model)))
        self.decoder = Encoder(EncoderLayer(args.d_model, c(attn), c(ff), dropout), args.N_dec)  # N_dec: encoder layers 复制次数
        self.oned_final_embed = torch.nn.Sequential(*[nn.Linear(args.d_model, 1)])  # make 32 channel to 1 channel. Convert to original channel
        self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)  # Convert 10(7+3) to 7(encoded codeword)

        self.get_mask(code)
        print(f'Mask:\n {self.src_mask}')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, magnitude, syndrome):
        magnitude = self.src_embed_m.unsqueeze(0) * magnitude.unsqueeze(-1)
        syndrome = self.src_embed_s.unsqueeze(0) * syndrome.unsqueeze(-1)
        magnitude, syndrome = self.decoder(magnitude, syndrome, self.src_mask)
        emb = torch.cat([magnitude, syndrome], -2)
        return self.out_fc(self.oned_final_embed(emb).squeeze(-1))

    def loss(self, z_pred, z2, y):
        loss = F.binary_cross_entropy_with_logits(
            z_pred, sign_to_bin(torch.sign(z2)))
        x_pred = sign_to_bin(torch.sign(-z_pred * torch.sign(y)))
        return loss, x_pred

    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask(code):
            mask = torch.tensor(code.pc_matrix)
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask
        src_mask = build_mask(code)
        self.register_buffer('src_mask', src_mask)
