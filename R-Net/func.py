import torch.nn as nn
import torch.nn.functional as F
import torch

INF = 1e30


class GRU(nn.Module):

    def __init__(self, num_layers, hidden_size, input_size, dropout=0.3, is_bidirectional=True, is_train=None):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.is_train = is_train
        # batch_first=True使得维度为[batch_size, seq_length, dim]，否则为[seq_length, batch_size, dim]
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        self.num_directions = 2 if is_bidirectional else 1  # 2表示双向

    def forward(self, x):
        x = self.dropout(x) if self.is_train else x
        # out：最后一层每个time-step的输出[batch_size, seq_length, hidden_size*num_directions]
        # h_n：保存每一层最后一个时间步隐状态：[batch, num_layers * num_directions, hidden_size]
        self.gru.flatten_parameters()
        out, h_n = self.gru(x)

        return out, h_n


class DotAttention(nn.Module):
    def __init__(self, hidden, dropout=0.3, is_train=True, input_dim=150, memory_dim=150, gated_dim=300):
        super(DotAttention, self).__init__()
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.is_train = is_train
        self.inputs_dense = Dense(input_dim, self.hidden)
        self.memory_dense = Dense(memory_dim, self.hidden)
        self.gated_dense = Dense(gated_dim, gated_dim)

    def forward(self, inputs, memory, mask):
        # 以Gated Attention-based Recurrent Networks这层维度为例
        # inputs：[N, PL, 150]
        # memory：[N, QL, 150]
        # q_mask：[N, QL]
        # hidden：75
        d_inputs = self.dropout(inputs) if self.is_train else inputs
        d_memory = self.dropout(memory) if self.is_train else memory
        JX = inputs.size(1)

        # attention
        inputs_ = F.relu(self.inputs_dense(d_inputs))  # [N, PL, 75]
        memory_ = F.relu(self.memory_dense(d_memory))  # [N, QL, 75]
        # 对篇章中的每个词，与整个问题中的每个词进行加权融合
        outputs = torch.matmul(inputs_, memory_.transpose(2, 1)) / (self.hidden ** 0.5)  # [N, PL, QL]
        mask = torch.unsqueeze(mask, 1).repeat(1, JX, 1).float()  # [N, PL, QL]
        softmax_mask = -INF * (1 - mask) + outputs
        logits = F.softmax(softmax_mask, dim=-1)  # [N, PL, QL]
        # 融合整个问题后的表示𝑐_𝑡
        outputs = torch.matmul(logits, memory)  # [N, PL, 150]
        # 将原表示与其连接：[u_𝑡^𝑃,𝑐_𝑡 ]
        res = torch.cat([inputs, outputs], dim=2)  # [N, PL, 300]

        # gates
        d_res = self.dropout(res) if self.is_train else res
        gate = torch.sigmoid(self.gated_dense(d_res))

        return gate * res  # [N, PL, 300]


class Dense(nn.Module):
    def __init__(self, input_size, hidden):
        super(Dense, self).__init__()
        self.hidden = hidden
        self.w = nn.Linear(input_size, hidden)

    def forward(self, inputs):
        # inputs：[N, maxlen, dim]
        shape = inputs.size()  # [N, maxlen, dim]
        out_shape = [shape[i] for i in range(len(shape) - 1)] + [self.hidden]  # [N, maxlen, hidden]

        flat_inputs = inputs.contiguous().view([-1, shape[-1]])  # [N*maxlen, dim]
        res = self.w(flat_inputs).view(out_shape)  # [N, maxlen, hidden]
        return res


class PtrNet(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3, is_train=None):
        super(PtrNet, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.is_train = is_train
        self.gru = GRU(1, hidden_size, input_size, dropout, is_bidirectional=False, is_train=is_train)
        self.pointer = Pointer(hidden_size)

    def forward(self, init, match, c_mask):
        d_match = self.dropout(match) if self.is_train else match
        d_init = self.dropout(init) if self.is_train else init
        # 开始位置概率分布
        # logits1：[N, PL]
        inp, logits1 = self.pointer(d_match, d_init, c_mask)
        d_inp = self.dropout(inp) if self.is_train else inp
        # 在经过一个时间步预测结束位置
        _, state = self.gru(d_inp)
        # 只取最后一层的输出
        state = self.dropout(state) if self.is_train else state
        state = state.squeeze(0)
        # 结束位置概率分布
        # logits2：[N, PL]
        _, logits2 = self.pointer(d_match, state, c_mask)

        return logits1, logits2


class Pointer(nn.Module):
    def __init__(self, hidden):
        super(Pointer, self).__init__()
        self.s0_dense = Dense(225, hidden)
        self.s1_dense = Dense(hidden, 1)

    def forward(self, inputs, state, c_mask):
        # inputs：[N, PL, 75]
        # state：[N, 150]
        # hidden：75
        # c_mask：[N, PL]
        u = torch.cat([state.unsqueeze(dim=1).repeat([1, inputs.size(1), 1]), inputs], dim=2)
        s0 = torch.tanh(self.s0_dense(u))  # [N, PL, hidden]
        s = self.s1_dense(s0).squeeze(2)  # [N, PL]
        s1 = -INF * (1 - c_mask.float()) + s
        a = F.softmax(s1, dim=-1).unsqueeze(dim=2)  # [N, PL, 1]
        res = torch.mul(inputs, a).sum(dim=1).unsqueeze(dim=1)  # [N, 75]
        return res, s1


class InitState(nn.Module):
    def __init__(self, hidden, dropout=0.3, is_train=True):
        super(InitState, self).__init__()
        self.is_train = is_train
        self.dropout = nn.Dropout(dropout)
        self.s0_dense = Dense(150, hidden)
        self.s1_dense = Dense(hidden, 1)

    def forward(self, memory, mask):
        # memory：[N, QL, 150]
        # q_mask：[N, QL]
        # hidden：75
        d_memory = self.dropout(memory) if self.is_train else memory  # [N, QL, 150]
        s0 = torch.tanh(self.s0_dense(d_memory))  # [N, QL, hidden]
        s = self.s1_dense(s0).squeeze(2)  # [N, QL]
        s1 = -INF * (1 - mask.float()) + s
        a = F.softmax(s1, dim=-1).unsqueeze(dim=2)  # [N, QL, 1]
        res = torch.mul(memory, a).sum(dim=1)  # [N, 150]
        return res
