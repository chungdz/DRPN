import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 200),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(200, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output

class TitleEncoder(nn.Module):
    def __init__(self, cfg):
        super(TitleEncoder, self).__init__()
        self.cfg = cfg
        # self.word_embedding = nn.Embedding(cfg.word_num, cfg.word_dim)
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.word_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.word_self_attend = SelfAttend(cfg.hidden_size)

        self.user_mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.pos_self_attend = SelfAttend(cfg.hidden_size)
        self.neg_self_attend = SelfAttend(cfg.hidden_size)

        self.dropout = nn.Dropout(cfg.dropout)
        self.word_layer_norm = nn.LayerNorm(cfg.hidden_size)
        self.user_layer_norm = nn.LayerNorm(cfg.hidden_size)

    def _extract_hidden_rep(self, seqs):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)

        # seq_masks = create_mask_from_lengths_for_seqs(seq_lens, self.max_news_len)
        # seq_masks = seq_masks.unsqueeze(1)
        # self_mask = seq_masks.transpose(-1, -2) * seq_masks
        # self_mask = self_mask.unsqueeze(1)
        #
        # trans_mask = (1.0 - self_mask.float()) * -10000.0
        # seq_h = self.encoder.forward(embs, trans_mask)
        X = self.dropout(embs)

        X = X.permute(1, 0, 2)
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)
        X = X.permute(1, 0, 2)
        # output = self.word_proj(output)

        return self.word_layer_norm(output + X)

    def encode_news(self, seqs):
        """

        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]

        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens)

        return self_attend

    def encode_user(self, seqs, utype):
        """

        Args:
            seqs: [*, max_hist_len, hidden_size]

        Returns:
            [*, hidden_size]
        """
        if utype == 'pos':
            user_mh_self_attn = self.user_mh_self_attn
            news_self_attend = self.pos_self_attend
        else:
            user_mh_self_attn = self.user_mh_self_attn
            news_self_attend = self.neg_self_attend

        hiddens = seqs.permute(1, 0, 2)
        user_hiddens, _ = user_mh_self_attn(hiddens, hiddens, hiddens)
        user_hiddens = user_hiddens.permute(1, 0, 2)

        residual_sum = self.user_layer_norm(user_hiddens + seqs)
        user_title_hidden = news_self_attend(residual_sum)

        return user_title_hidden


class NodesEncoder(nn.Module):
    def __init__(self, cfg):
        super(NodesEncoder, self).__init__()
        self.cfg = cfg
        
        self.user_mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.nodes_mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.pos_cross_attend = CrossAttend(cfg.hidden_size, cfg.pos_hist_length)
        self.neg_cross_attend = CrossAttend(cfg.hidden_size, cfg.neg_hist_length)
        self.pos_self_attend = SelfAttend(cfg.hidden_size)
        self.neg_self_attend = SelfAttend(cfg.hidden_size)
        self.pos_sca = SelfCrossAttend(cfg)
        self.neg_sca = SelfCrossAttend(cfg)

        self.dropout = nn.Dropout(cfg.dropout)
        self.user_layer_norm = nn.LayerNorm(cfg.hidden_size)
        self.pos_weight = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, 1),
            nn.Sigmoid()
        )
        self.neg_weight = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, 1),
            nn.Sigmoid()
        )

        self.pos_cross_nodes = CrossAttend(cfg.hidden_size, cfg.pos_hist_length)
        self.neg_cross_nodes = CrossAttend(cfg.hidden_size, cfg.neg_hist_length)
        self.pos_self_nodes = SelfAttend(cfg.hidden_size)
        self.neg_self_nodes = SelfAttend(cfg.hidden_size)
        self.pos_sca_nodes = SelfCrossAttend(cfg)
        self.neg_sca_nodes = SelfCrossAttend(cfg)

    def forward(self, pos, neg, pos_nodes, neg_nodes):
        """

        Args:
            seqs: [*, max_hist_len, hidden_size]

        Returns:
            [*, hidden_size]
        """

        pos_permuted = pos.permute(1, 0, 2)
        pos_hiddens, _ = self.user_mh_self_attn(pos_permuted, pos_permuted, pos_permuted)
        pos_hiddens = pos_hiddens.permute(1, 0, 2)
        pos_residual = self.user_layer_norm(pos_hiddens + pos)

        neg_permuted = neg.permute(1, 0, 2)
        neg_hiddens, _ = self.user_mh_self_attn(neg_permuted, neg_permuted, neg_permuted)
        neg_hiddens = neg_hiddens.permute(1, 0, 2)
        neg_residual = self.user_layer_norm(neg_hiddens + neg)

        pos_s = self.pos_self_attend(pos_residual)
        neg_s = self.neg_self_attend(neg_residual)

        pos_self, pos_cross = self.pos_cross_attend(pos, neg)
        neg_self, neg_cross = self.neg_cross_attend(neg, pos)

        pos_c = self.pos_sca(pos, pos_self, pos_cross)
        neg_c = self.neg_sca(neg, neg_self, neg_cross)

        # pos_nodes_permuted = pos_nodes.permute(1, 0, 2)
        # pos_nodes_hiddens, _ = self.user_mh_self_attn(pos_nodes_permuted, pos_nodes_permuted, pos_nodes_permuted)
        # pos_nodes_hiddens = pos_nodes_hiddens.permute(1, 0, 2)
        # pos_nodes_residual = self.user_layer_norm(pos_nodes_hiddens + pos_nodes)

        # neg_nodes_permuted = neg_nodes.permute(1, 0, 2)
        # neg_nodes_hiddens, _ = self.user_mh_self_attn(neg_nodes_permuted, neg_nodes_permuted, neg_nodes_permuted)
        # neg_nodes_hiddens = neg_nodes_hiddens.permute(1, 0, 2)
        # neg_nodes_residual = self.user_layer_norm(neg_nodes_hiddens + neg_nodes)

        pos_s_nodes = self.pos_self_attend(pos_nodes)
        neg_s_nodes = self.neg_self_attend(neg_nodes)

        pos_self_nodes, pos_cross_nodes = self.pos_cross_nodes(pos_nodes, neg_nodes)
        neg_self_nodes, neg_cross_nodes = self.neg_cross_nodes(neg_nodes, pos_nodes)

        pos_c_nodes = self.pos_sca_nodes(pos_nodes, pos_self_nodes, pos_cross_nodes)
        neg_c_nodes = self.neg_sca_nodes(neg_nodes, neg_self_nodes, neg_cross_nodes)

        return pos_s, pos_s_nodes, neg_s, neg_s_nodes, pos_c, pos_c_nodes, neg_c, neg_c_nodes
        # return pos_s, neg_s

class SelfCrossAttend(nn.Module):
    def __init__(self, cfg) -> None:
        super(SelfCrossAttend, self).__init__()

        self.alpha = nn.Parameter(torch.Tensor([0.5]))
        self.w = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1)
        )

    def forward(self, q, k1, k2):
        # q (batch_size, seq_len, hidden_size)
        # k1 (batch_size, seq_len, hidden_size)
        # k2 (batch_size, seq_len, hidden_size)

        sim1 = self.w(torch.cat([q, k1], dim=-1))
        sim2 = self.w(torch.cat([q, k2], dim=-1))
        
        # sim batch_size, seq_len, 1
        sim = torch.softmax(sim1 - torch.relu(self.alpha) * sim2, dim=1)
        output = (q * sim).sum(dim=1)

        return output

class MaskedSelfAttend(nn.Module):
    def __init__(self, hidden_size, mask_len) -> None:
        super(MaskedSelfAttend, self).__init__()

        # self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.mask = nn.Parameter(torch.eye(mask_len) == 1, requires_grad=False)
        self.hidden_size = hidden_size

    def forward(self, q):
        # q (batch_size, seq_len, hidden_size)
        
        k = q.permute(0, 2, 1)
        sim = torch.matmul(q, k) / math.sqrt(self.hidden_size)
        sim = torch.softmax(sim.masked_fill_(self.mask, -1e9), dim=-1)
        output = torch.matmul(sim, q)

        return output

class CrossAttend(nn.Module):
    def __init__(self, hidden_size, mask_len) -> None:
        super(CrossAttend, self).__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.hidden_size = hidden_size
        self.mask = nn.Parameter(torch.eye(mask_len) == 1, requires_grad=False)

    def forward(self, q, opp):
        # q (batch_size, seq_len, hidden_size)
        # v (batch_size, seq_len_k, hidden_size)
        q = self.query(q)
        k = self.key(q)
        v = self.value(q)

        k = k.permute(0, 2, 1)
        sim = torch.matmul(q, k) / math.sqrt(self.hidden_size)
        sim = torch.softmax(sim.masked_fill_(self.mask, -1e9), dim=-1)
        self_output = torch.matmul(sim, v)

        opp_k = self.key(opp).permute(0, 2, 1)
        opp_v = self.value(opp)

        sim2 = torch.matmul(q, opp_k) / math.sqrt(self.hidden_size)
        # sim (batch_size, seq_len, seq_len_k)
        sim2 = torch.softmax(sim2, dim=-1)
        opp_output = torch.matmul(sim2, opp_v)
        # output (batch_size, seq_len, hidden_size)

        return self_output, opp_output

class Multihead_bandti(nn.Module):

    def __init__(self, cfg):

        super(Multihead_bandti, self).__init__()

        self.head_num = cfg.head_num
        self.head_dim = cfg.hidden_size // cfg.head_num
        self.hidden_size = cfg.hidden_size
        
        self.policy_1 = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, self.head_num)
        )
        self.policy_2 = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, self.head_num)
        )
        self.policy_3 = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, self.head_num)
        )
        self.policy_4 = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, self.head_num)
        )
    
    def forward(self, refer, s1, s2, s3, s4):
        # refer batch_size, neg_num + 1, hidden_size
        # s batch_size, neg_num + 1, hidden_size

        gamma_1 = self.policy_1(refer).unsqueeze(-1)
        gamma_2 = self.policy_2(refer).unsqueeze(-1)
        gamma_3 = self.policy_3(refer).unsqueeze(-1)
        gamma_4 = self.policy_4(refer).unsqueeze(-1)
        # gamma batch_size, neg_num + 1, head_num, 1

        s1 = s1.view(-1, refer.size(1), self.head_num, self.head_dim)
        s2 = s2.view(-1, refer.size(1), self.head_num, self.head_dim)
        s3 = s3.view(-1, refer.size(1), self.head_num, self.head_dim)
        s4 = s4.view(-1, refer.size(1), self.head_num, self.head_dim)

        final = gamma_1 * s1 + gamma_2 * s2 + gamma_3 * s3 + gamma_4 * s4
        final = final.reshape(-1, refer.size(1), self.hidden_size)

        return final

