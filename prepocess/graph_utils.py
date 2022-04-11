import os
import json
import pickle
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import argparse

def build_entire_graph(news_list, user_dict):
    edges = list()

    for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='build graph'):
        seq = uinfo['pos']
        cur_len = len(seq)
        
        if cur_len > 1:
            
            for i in range(1, cur_len):
                edges.append([seq[i], seq[i - 1], 1])
                edges.append([seq[i - 1], seq[i], 1])

    edge_df = pd.DataFrame(edges, columns=["from", "to", "weight"])
    edge_weights = edge_df.groupby(["from", "to"]).apply(lambda x: sum(x["weight"]))
    weighted_edges = edge_weights.to_frame().reset_index().values

    dg = nx.DiGraph()
    dg.add_nodes_from(news_list)
    dg.add_weighted_edges_from(weighted_edges)

    return dg

def find_news_neighbor(news_dict, dg, max_neighbor_cnt=5):

    not_found = 0
    neighbor_dict = {}
    for nid, ninfo in tqdm(news_dict.items(), total=len(news_dict), desc='find news neighbor'):

        idx = ninfo['idx']
        neighbor_dict[idx] = []
            
        sorted_neighbors = sorted(dict(dg[idx]).items(), key=lambda item: -1 * item[1]['weight'])
        if len(sorted_neighbors) < 1:
            not_found += 1
            continue
        neighbors = [int(x[0]) for x in sorted_neighbors][:max_neighbor_cnt]

        neighbor_dict[idx] = neighbors

    print('There are {} news has no neighbors'.format(not_found))
    return neighbor_dict

def find_user_neighbor(neighbor_dict, user_dict, dg, news_dict, neighbor_news_num=90):

    for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='user neighbor'):
        # integrate neighbors
        user_news = set(uinfo['pos'] + uinfo['neg'])
        neighbor_news = set()
        edges = set()
        for nid in user_news:
            for neighbor_nid in neighbor_dict[nid]:
                if neighbor_nid not in user_news:
                    neighbor_news.add(neighbor_nid)
                edges.add((nid, neighbor_nid))
                edges.add((neighbor_nid, nid))
        # build subgraph
        neighbor_list = list(neighbor_news)
        neighbor_len = len(neighbor_list)
        if neighbor_len < neighbor_news_num:
            for _ in range(neighbor_news_num - neighbor_len):
                neighbor_list.append(news_dict['<his>']['idx'])
        uinfo['all_nodes'] = uinfo['pos'] + uinfo['neg'] + neighbor_list[:neighbor_news_num]
        assert(len(uinfo['all_nodes']) == 180)
        # sub_graph = dg.subgraph(uinfo['all_nodes'])

        sub_graph_node_idx = {news_idx: sub_idx for sub_idx, news_idx in enumerate(uinfo['all_nodes'])}
        # sub_graph = nx.relabel_nodes(sub_graph, sub_graph_node_idx)
        # find edge
        # edges = list(sub_graph.edges)
        source_nodes, target_nodes = [], []
        for edge in edges:
            if edge[0] not in sub_graph_node_idx or edge[1] not in sub_graph_node_idx:
                continue
            source_nodes.append(sub_graph_node_idx[edge[0]])
            target_nodes.append(sub_graph_node_idx[edge[1]])
        uinfo['edge_index'] = [source_nodes, target_nodes]
        # mask
        uinfo['pos_mask'] = [1 if x in uinfo['pos'] else 0 for x in uinfo['all_nodes']]
        uinfo['neg_mask'] = [1 if x in uinfo['neg'] else 0 for x in uinfo['all_nodes']]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='data', type=str)
    args = parser.parse_args()
    root = args.root

    dg = pickle.load(open('{}/graph.bin'.format(args.root), 'rb'))
    user_dict = json.load(open('{}/user.json'.format(args.root), 'r', encoding='utf-8'))
    news_dict = json.load(open('{}/news.json'.format(args.root), 'r', encoding='utf-8'))

    neighbor_dict = find_news_neighbor(news_dict, dg)
    find_user_neighbor(neighbor_dict, user_dict, dg, news_dict)

    json.dump(user_dict, open('{}/user_n.json'.format(args.root), 'w', encoding='utf-8'))


