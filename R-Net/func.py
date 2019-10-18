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
        # batch_first=True使得维度为[batch_size, seq_length, dim]
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        self.num_directions = 2 if is_bidirectional else 1  # 2表示双向

    def forward(self, x):
        # 初始化隐状态
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size)

        x = self.dropout(x) if self.is_train else x
        # out：[batch_size, seq_length, hidden_size*num_directions]
        # h_n：保存每一层最后一个时间步隐状态：[batch, num_layers * num_directions, hidden_size]
        out, h_n = self.gru(x, h0)

        # 返回最后一层每个time-step的输出
        return out, h_n


class DotAttention(nn.Module):
    def __init__(self, hidden, dropout=0.3, is_train=True):
        super(DotAttention, self).__init__()
        self.hidden = hidden
        self.dropout = nn.Dropout(dropout)
        self.is_train = is_train
        self.dense = Dense()

    def forward(self, inputs, memory, mask):
        # 以Gated Attention-based Recurrent Networks这层维度为例
        # inputs：[N, PL, 450]
        # memory：[N, QL, 450]
        # q_mask：[N, QL]
        # hidden：75
        d_inputs = self.dropout(inputs) if self.is_train else inputs
        d_memory = self.dropout(memory) if self.is_train else memory
        JX = inputs.size(1)

        # attention
        inputs_ = F.relu(self.dense(d_inputs, self.hidden))  # [N, PL, 75]
        memory_ = F.relu(self.dense(d_memory, self.hidden))  # [N, QL, 75]
        # 对篇章中的每个词，与整个问题中的每个词进行加权融合
        outputs = torch.matmul(inputs_, memory_.transpose(2, 1)) / (self.hidden ** 0.5)  # [N, PL, QL]
        mask = torch.unsqueeze(mask, 1).repeat(1, JX, 1).float()  # [N, PL, QL]
        softmax_mask = -INF * (1 - mask) + outputs
        logits = F.softmax(softmax_mask, dim=-1)  # [N, PL, QL]
        # 融合整个问题后的表示𝑐_𝑡
        outputs = torch.matmul(logits, memory)  # [N, PL, 450]
        # 将原表示与其连接：[u_𝑡^𝑃,𝑐_𝑡 ]
        res = torch.cat([inputs, outputs], dim=2)  # [N, PL, 900]

        # gates
        d_res = self.dropout(res) if self.is_train else res
        gate = torch.sigmoid(self.dense(d_res, res.size(-1)))

        return gate * res


class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.w = None

    def forward(self, inputs, hidden):
        # inputs：[N, maxlen, dim]
        shape = inputs.size()  # [N, maxlen, dim]
        self.w = nn.Linear(shape[-1], hidden)
        out_shape = [shape[i] for i in range(len(shape) - 1)] + [hidden]  # [N, maxlen, hidden]

        flat_inputs = inputs.view([-1, shape[-1]])  # [N*maxlen, dim]
        res = self.w(flat_inputs).view(out_shape)  # [N, maxlen, hidden]
        return res


class PtrNet(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.3, is_train=None):
        super(PtrNet).__init__()
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
        last_state = state[:, -1, :]
        d_state = self.dropout(last_state) if self.is_train else last_state
        # 结束位置概率分布
        # logits2：[N, PL]
        _, logits2 = self.pointer(d_match, d_state, c_mask)

        return logits1, logits2


class Pointer(nn.Module):
    def __init__(self, hidden):
        super(Pointer, self).__init__()
        self.hidden = hidden
        self.dense = Dense()

    def forward(self, inputs, state, c_mask):
        # inputs：[N, PL, 1800]
        # state：[N, 300]
        # hidden：75
        # c_mask：[N, PL]
        u = torch.cat([state.unsqueeze(dim=1).repeat([1, inputs.size(1), 1]), inputs], dim=2)
        s0 = F.tanh(self.dense(u, self.hidden))  # [N, PL, hidden]
        s = self.dense(s0, 1).squeeze(2)  # [N, PL]
        s1 = -INF * (1 - c_mask.float()) + s
        a = F.softmax(s1, dim=-1).unsqueeze(dim=2)  # [N, PL, 1]
        res = torch.matmul(inputs, a).sum(dim=1)  # [N, 1800]

        return res, s1


class InitState(nn.Module):
    def __init__(self, dropout=0.3, is_train=True):
        super(InitState, self).__init__()
        self.is_train = is_train
        self.dropout = nn.Dropout(dropout)
        self.dense = Dense()

    def forward(self, memory, hidden, mask):
        # memory：[N, QL, -2*hidden:]
        # q_mask：[N, QL]
        # hidden：75
        d_memory = self.dropout(memory) if self.is_train else memory  # [N, QL, 300]
        s0 = F.tanh(self.dense(d_memory, hidden))  # [N, QL, hidden]
        s = self.dense(s0, 1).squeeze(2)  # [N, QL]
        s1 = -INF * (1 - mask.float()) + s
        a = F.softmax(s1, dim=-1).unsqueeze(dim=2)  # [N, QL, 1]
        res = torch.matmul(memory, a).sum(dim=1)  # [N, 300]
        return res
