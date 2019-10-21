import torch.nn as nn
import torch
from func import GRU, DotAttention, InitState, PtrNet


class RNET(nn.Module):
    def __init__(self, config, device, is_train=True, word_mat=None, char_mat=None, opt=True):
        super(RNET, self).__init__()
        self.config = config
        self.device = device
        self.is_train = is_train
        self.opt = opt
        self.emb = Embedding(config, is_train, word_mat, char_mat)
        self.encode = Encoder(config, is_train)
        self.gatedAtt = GatedAttention(config, is_train)
        self.selfMatch = SelfMatch(config, is_train)
        self.answerOutput = AnswerOutput(config, is_train)

    def forward(self, c, q, ch, qh):
        '''
        传进来的参数都是填充后的
        :param c: 上下文词矩阵[batch_size,para_limit]
        :param q: 问题词矩阵[batch_size,ques_limit]
        :param ch: 上下文字符矩阵[batch_size,para_limit,char_limit]
        :param qh: 问题字符矩阵[batch_size,ques_limit,char_limit]
        :return:
        '''

        # 计算mask矩阵,填充部分不计算在内
        c_mask = c.bool()  # [batch_size,para_limit]
        q_mask = q.bool()  # [batch_size,ques_limit]
        # 计算上下文/问题长度
        c_len = c_mask.int().sum(dim=1)  # [batch_size]
        q_len = q_mask.int().sum(dim=1)  # [batch_size]

        if self.opt:
            N, CL = c.size(0), self.config.char_limit
            c_maxlen = c_len.max().tolist()
            q_maxlen = q_len.max().tolist()
            # 取最大长度的上下文/问题/答案
            # 未作嵌入前
            c = c[:, :c_maxlen]
            q = q[:, :q_maxlen]
            c_mask = c_mask[:, :c_maxlen]
            q_mask = q_mask[:, :q_maxlen]
            ch = ch[:, :c_maxlen, :CL]
            qh = qh[:, :q_maxlen, :CL]
        else:
            c_maxlen, q_maxlen = self.config.para_limit, self.config.ques_limit

        # 上下文/问题中单词的长度
        ch_len = ch.bool().int().sum(dim=2).view(-1).tolist()
        qh_len = qh.bool().int().sum(dim=2).view(-1).tolist()

        # 嵌入
        c_emb, q_emb = self.emb(c, q, ch, qh, c_maxlen, q_maxlen)
        # 编码
        c, q = self.encode(c_emb, q_emb)
        # 得到问题感知的每个篇章词新的表示
        c_att = self.gatedAtt(c, q, q_mask)
        # 得到融合了上下文的每个篇章词新的表示
        match = self.selfMatch(c_att, c_mask)
        # 开始和结束位置概率分布：[N, PL]
        logits1, logits2 = self.answerOutput(q, match, c_mask, q_mask)

        return logits1, logits2


class Embedding(nn.Module):
    def __init__(self, config, is_train=True, word_mat=None, char_mat=None):
        super(Embedding, self).__init__()
        self.config = config
        self.is_train = is_train
        self.dropout = nn.Dropout(config.drop_prob)
        # 使用预训练的glove词向量，不微调
        self.word_mat = nn.Embedding.from_pretrained(word_mat, freeze=True)
        self.char_mat = nn.Embedding.from_pretrained(char_mat, freeze=False)
        self.gru = GRU(1, config.char_hidden, config.char_dim, config.drop_prob, is_bidirectional=True,
                       is_train=is_train)

    def forward(self, c, q, ch, qh, c_maxlen, q_maxlen):
        N, PL, QL, CL, dc, dg = c.size(
            0), c_maxlen, q_maxlen, self.config.char_limit, self.config.char_dim, self.config.char_hidden
        # 字符嵌入
        ch_emb = self.char_mat(ch.long()).view([N * PL, CL, dc])
        qh_emb = self.char_mat(qh.long()).view([N * QL, CL, dc])
        ch_emb = self.dropout(ch_emb) if self.is_train else ch_emb
        qh_emb = self.dropout(qh_emb) if self.is_train else qh_emb
        # 双向rnn最后一个隐状态表示每个词
        _, ch_emb = self.gru(ch_emb)
        _, qh_emb = self.gru(qh_emb)
        # 经过rnn得到上下文/问题中每个词的字符嵌入
        ch_emb = ch_emb.view([N, PL, 2 * dg])
        qh_emb = qh_emb.view([N, QL, 2 * dg])

        # 词嵌入
        c_emb = self.word_mat(c.long())  # [N, PL, 300]
        q_emb = self.word_mat(q.long())  # [N, QL, 300]

        # 连接每个词的词嵌入和字符嵌入，获得新的嵌入表示
        c_emb = torch.cat([c_emb, ch_emb], dim=2)  # [N, PL, 300+2*dg]
        q_emb = torch.cat([q_emb, qh_emb], dim=2)  # [N, QL, 300+2*dg]

        return c_emb, q_emb


class Encoder(nn.Module):
    def __init__(self, config, is_train=True):
        super(Encoder, self).__init__()
        self.config = config
        self.is_train = is_train
        self.gru = None

    def forward(self, c_emb, q_emb):
        input_size = c_emb.size(-1)
        self.gru = GRU(3, self.config.hidden, input_size, self.config.drop_prob, is_bidirectional=True,
                       is_train=self.is_train)

        c, _ = self.gru(c_emb)
        q, _ = self.gru(q_emb)

        return c, q


class GatedAttention(nn.Module):
    def __init__(self, config, is_train=True):
        super(GatedAttention, self).__init__()
        self.config = config
        self.is_train = is_train
        self.dot_attention = DotAttention(config.hidden, config.drop_prob, is_train)
        self.gru = None

    def forward(self, c, q, q_mask):
        qc_att = self.dot_attention(c, q, q_mask)

        self.gru = GRU(1, self.config.hidden, qc_att.size(-1), dropout=self.config.drop_prob, is_bidirectional=False,
                       is_train=self.is_train)
        c_att, _ = self.gru(qc_att)

        return c_att


class SelfMatch(nn.Module):
    def __init__(self, config, is_train=True):
        super(SelfMatch, self).__init__()
        self.config = config
        self.is_train = is_train
        self.dot_attention = DotAttention(config.hidden, config.drop_prob, is_train)
        self.gru = None

    def forward(self, c_att, c_mask):
        qc_att = self.dot_attention(c_att, c_att, c_mask)

        self.gru = GRU(1, self.config.hidden, qc_att.size(-1), dropout=self.config.drop_prob, is_bidirectional=False,
                       is_train=self.is_train)
        match, _ = self.gru(qc_att)

        return match


class AnswerOutput(nn.Module):
    def __init__(self, config, is_train=True):
        super(AnswerOutput, self).__init__()
        self.config = config
        self.is_train = is_train
        self.init_state = InitState(self.config.ptr_drop_prob, is_train)
        self.pointer = None

    def forward(self, q, match, c_mask, q_mask):
        init = self.init_state(q[:, :, -2 * self.config.hidden:], self.config.hidden, q_mask)
        self.pointer = PtrNet(match.size(-1), init.size(-1), self.config.ptr_drop_prob, is_train=self.is_train)
        # 开始和结束位置概率分布：[N, PL]
        logits1, logits2 = self.pointer(init, match, c_mask)

        return logits1, logits2
