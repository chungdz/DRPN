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
import time

random.seed(7)

def build_examples(rank, args, df, news_title, news_info, user_info, fout):
    group = rank // 5
    time.sleep(300 * group) 

    data_list = []
    for imp_id, uid, imp in tqdm(df[["id", "uid", "imp"]].values, total=df.shape[0]):
        if uid not in user_info:
            continue

        imp_list = str(imp).split(' ')
        for impre in imp_list:
            arr = impre.split('-')
            curn = news_info[arr[0]]['idx']
            label = int(arr[1])

            new_row = convert_torch(imp_id, label, user_info[uid], news_title, curn)
            
            data_list.append(new_row)
    
    data, slices = RecoData.collate(data_list)
    torch.save((data, slices), fout)

def convert_torch(imp_id, label, uinfo, news_info, target_news_id):
    
    # build Data
    x = torch.LongTensor(uinfo['all_nodes']).unsqueeze(1)
    y = torch.LongTensor([label])
    edge_index = torch.LongTensor(uinfo['edge_index'])
    curdata = Data(x=x, edge_index=edge_index, y=y)
    # other type
    curdata.imp_id = torch.LongTensor([imp_id, ])
    curdata.target_news_id = torch.LongTensor([target_news_id, ]).unsqueeze(0)
    pos_news_info = []
    for pos_id in uinfo['pos']:
        pos_news_info += list(news_info[pos_id])
    curdata.pos_news_info = torch.LongTensor(pos_news_info).unsqueeze(0)
    neg_news_info = []
    for neg_id in uinfo['neg']:
        neg_news_info += list(news_info[neg_id])
    curdata.neg_news_info = torch.LongTensor(neg_news_info).unsqueeze(0)
    target_news_info = list(news_info[target_news_id])
    curdata.target_info = torch.LongTensor(target_news_info).unsqueeze(0)
    curdata.pos_mask = torch.arange(30)
    curdata.neg_mask = torch.arange(30, 90)
    
    return curdata

def main(args):
    f_dev_beh = os.path.join("data", args.fsamples)
    df = pd.read_csv(f_dev_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    news_title = np.load('data/news_info.npy')
    news_info = json.load(open('data/news.json', 'r', encoding='utf-8'))
    user_info = json.load(open('data/user_n.json', 'r', encoding='utf-8'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join("data", args.fout,  "dev-{}.pt".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_title, news_info, user_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="dev/target_behaviors.tsv", type=str,
                        help="Path of the deving samples file.")
    parser.add_argument("--fout", default="processed", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--neg_count", default=4, type=int)
    parser.add_argument("--processes", default=10, type=int, help="Processes number")

    args = parser.parse_args()

    main(args)

