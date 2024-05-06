import torch
import torch.nn.functional as F

# 参数
batch_size = 10
seq_length = 1
d_model = 32
num_heads = 8
d_k = d_model // num_heads  # 每个头的维度

# 输入数据和掩码
x = torch.randn(batch_size, seq_length, d_model)  # [10, 1, 32]
mask = torch.ones(seq_length, seq_length).tril()  # 下三角矩阵表示未来时刻掩码 [10, 10]

# 初始化线性变换层
linear_q = torch.nn.Linear(d_model, num_heads * d_k)
linear_k = torch.nn.Linear(d_model, num_heads * d_k)
linear_v = torch.nn.Linear(d_model, num_heads * d_k)

# 执行线性变换
q0 = linear_q(x)  # [10, 10, 32]
k0 = linear_k(x)  # [10, 10, 32]
v0 = linear_v(x)  # [10, 10, 32]

# 将查询、键、值变形为多个头
q = q0.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)  # [10, 8, 10, 4]
k = k0.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)  # [10, 8, 10, 4]
v = v0.view(batch_size, seq_length, num_heads, d_k).transpose(1, 2)  # [10, 8, 10, 4]

# 扩展掩码以适应多头注意力
mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, 10, 10]
mask = mask.expand(batch_size, num_heads, seq_length, seq_length)  # [1000, 8, 10, 10]

# 自注意力机制
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # [10, 8, 10, 10]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 掩码填充
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return torch.matmul(attn, value), attn  # [10, 8, 10, 4], [1000, 8, 10, 10]

# 计算注意力输出
output, attention_weights = attention(q, k, v, mask)

print("查询形状：", q.shape)  # [1000, 8, 10, 4]
print("键形状：", k.shape)    # [1000, 8, 10, 4]
print("值形状：", v.shape)    # [1000, 8, 10, 4]
print("掩码形状：", mask.shape)  # [1000, 8, 10, 10]
print("注意力权重形状：", attention_weights.shape)  # [1000, 8, 10, 10]
