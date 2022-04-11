import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import TitleEncoder, SelfAttend, NodesEncoder
from models.gat import GatNet
from models.graph_pooling import MaskGlobalAttention, GateLayer

class PNRec(nn.Module):
    def __init__(self, cfg):
        super(PNRec, self).__init__()

        self.title_encoder = TitleEncoder(cfg)
        # self.nodes_encoder = NodesEncoder(cfg)
        self.news_encoder = NodesEncoder(cfg)
        self.cfg = cfg
        self.policy_pos_s = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.policy_neg_s = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.policy_pos_c = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.policy_neg_c = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )

        self.policy_pos_s_nodes = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.policy_neg_s_nodes = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.policy_pos_c_nodes = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.policy_neg_c_nodes = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),
        )

        self.news_embedding = nn.Embedding(cfg.news_num, cfg.hidden_size)
        self.gnn = GatNet(cfg.hidden_size, cfg.hidden_size)

        self.title_self_attend = SelfAttend(cfg.hidden_size)
        self.node_self_attend = SelfAttend(cfg.hidden_size)

    def forward(self, data, test_mode=False):
        neg_num = self.cfg.neg_count
        if test_mode:
            neg_num = 0

        target_news = data.target_info.reshape(-1, self.cfg.max_title_len)
        target_news = self.title_encoder.encode_news(target_news).reshape(-1, neg_num + 1, self.cfg.hidden_size)
        target_nodes = self.news_embedding(data.target_news_id)
        target_all = torch.cat([target_news, target_nodes], dim=-1)

        pos_his = data.pos_news_info.reshape(-1, self.cfg.max_title_len)
        pos_his = self.title_encoder.encode_news(pos_his).reshape(-1, self.cfg.pos_hist_length, self.cfg.hidden_size)

        neg_his = data.neg_news_info.reshape(-1, self.cfg.max_title_len)
        neg_his = self.title_encoder.encode_news(neg_his).reshape(-1, self.cfg.neg_hist_length, self.cfg.hidden_size)

        title_v = self.title_self_attend(torch.cat([pos_his, neg_his], dim=-2))
        title_v = title_v.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        # graph
        nodes = self.news_embedding(data.x.squeeze(1))
        node_v = self.node_self_attend(
            nodes.reshape(-1, self.cfg.pos_hist_length + self.cfg.neg_hist_length + self.cfg.neighbor_length, self.cfg.hidden_size)[:, :self.cfg.pos_hist_length + self.cfg.neg_hist_length, :])
        node_v = node_v.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)

        nodes = self.gnn(nodes, data.edge_index).reshape(-1, self.cfg.pos_hist_length + self.cfg.neg_hist_length + self.cfg.neighbor_length, self.cfg.hidden_size)
        pos_nodes = nodes[:, :self.cfg.pos_hist_length, :]
        neg_nodes = nodes[:, self.cfg.pos_hist_length: self.cfg.pos_hist_length + self.cfg.neg_hist_length, :]
        # pos_nodes, neg_nodes = self.nodes_encoder(pos_nodes, neg_nodes)

        pos_s, pos_s_nodes, neg_s, neg_s_nodes, pos_c, pos_c_nodes, neg_c, neg_c_nodes = self.news_encoder(pos_his, neg_his, pos_nodes, neg_nodes)
        
        pos_s = pos_s.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        pos_s_nodes = pos_s_nodes.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        neg_s = neg_s.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        neg_s_nodes = neg_s_nodes.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        pos_c = pos_c.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        pos_c_nodes = pos_c_nodes.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        neg_c = neg_c.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        neg_c_nodes = neg_c_nodes.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)

        news_states = torch.cat([title_v, target_news], dim=-1)
        gamma_1 = self.policy_pos_s(news_states)
        gamma_2 = self.policy_neg_s(news_states)
        gamma_3 = self.policy_pos_c(news_states)
        gamma_4 = self.policy_neg_c(news_states)

        news_final = gamma_1 * pos_s + gamma_2 * neg_s + gamma_3 * pos_c + gamma_4 * neg_c

        node_states = torch.cat([node_v, target_nodes], dim=-1)
        gamma_node_1 = self.policy_pos_s_nodes(node_states)
        gamma_node_2 = self.policy_neg_s_nodes(node_states)
        gamma_node_3 = self.policy_pos_c_nodes(node_states)
        gamma_node_4 = self.policy_neg_c_nodes(node_states)

        node_final = gamma_node_1 * pos_s_nodes + gamma_node_2 * neg_s_nodes + gamma_node_3 * pos_c_nodes + gamma_node_4 * neg_c_nodes

        return torch.sum(torch.cat([news_final, node_final], dim=-1) * target_all, dim=-1)