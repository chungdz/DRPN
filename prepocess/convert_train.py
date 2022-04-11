import os
import json
import pickle
import argparse
import math
import pandas as pd
import numpy as np
import multiprocessing as mp
import random
from tqdm import tqdm
from datasets.recodata import RecoData
from torch_geometric.data import Data
import torch

random.seed(7)

def build_examples(rank, args, news_info, user_info, fout, file_num):
    np_list = []
    for i in range(rank * file_num, (rank + 1) * file_num):
        output_path = os.path.join(args.root, "raw",  "train-{}.npy".format(i))
        np_list.append(np.load(output_path))
    datanp = np.concatenate(np_list, axis=0)

    data_list = []
    for info_list in tqdm(datanp, total=datanp.shape[0]):
        # str to int
        new_row = []
        for t in range(1 + 1 + 1 + args.neg_count):
            new_row.append(int(info_list[t]))
        # user info
        uinfo = user_info[info_list[-1]]
        # build Data
        x = torch.LongTensor(uinfo['all_nodes']).unsqueeze(1)
        y = torch.LongTensor([0])
        edge_index = torch.LongTensor(uinfo['edge_index'])
        curdata = Data(x=x, edge_index=edge_index, y=y)
        # other type
        curdata.imp_id = torch.LongTensor([new_row[0], ])
        curdata.target_news_id = torch.LongTensor(new_row[-5:]).unsqueeze(0)
        pos_news_info = []
        for pos_id in uinfo['pos']:
            pos_news_info += list(news_info[pos_id])
        curdata.pos_news_info = torch.LongTensor(pos_news_info).unsqueeze(0)
        neg_news_info = []
        for neg_id in uinfo['neg']:
            neg_news_info += list(news_info[neg_id])
        curdata.neg_news_info = torch.LongTensor(neg_news_info).unsqueeze(0)
        target_news_info = []
        for target_id in new_row[-5:]:
            target_news_info += list(news_info[target_id])
        curdata.target_info = torch.LongTensor(target_news_info).unsqueeze(0)
        curdata.pos_mask = torch.arange(30)
        curdata.neg_mask = torch.arange(30, 90)
        
        data_list.append(curdata)
    
    data, slices = RecoData.collate(data_list)
    torch.save((data, slices), fout)

def main(args):
    
    news_info = np.load('{}/news_info.npy'.format(args.root))
    user_info = json.load(open('{}/user_n.json'.format(args.root), 'r', encoding='utf-8'))
    file_num = 8 // args.processes
    processes = []
    for i in range(args.processes):
        output_path = os.path.join(args.root, args.fout,  "train-{}.pt".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, news_info, user_info, output_path, file_num))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fout", default="processed", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--neg_count", default=4, type=int)
    parser.add_argument("--processes", default=4, type=int, help="Processes number")
    parser.add_argument("--root", default="data", type=str)
    args = parser.parse_args()

    main(args)

