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

random.seed(7)

def build_examples(rank, args, df, news_info, user_info, fout):
    data_list = []
    for imp_id, uid, imp in tqdm(df[["id", "uid", "imp"]].values, total=df.shape[0]):
        if uid not in user_info:
            continue

        imp_list = str(imp).split(' ')
        imp_pos_list = []
        imp_neg_list = []
        for impre in imp_list:
            arr = impre.split('-')
            curn = news_info[arr[0]]['idx']
            label = int(arr[1])

            if label == 0:
                imp_neg_list.append(curn)
            elif label == 1:
                imp_pos_list.append(curn)
            else:
                raise Exception('label error!')
        
        neg_num = len(imp_neg_list)
        if neg_num < args.neg_count:
            for i in range(args.neg_count - neg_num):
                imp_neg_list.append(news_info['<pad>']['idx'])
        
        for p in imp_pos_list:
            sampled = random.sample(imp_neg_list, args.neg_count)
            new_row = []
            new_row.append(int(imp_id))
            new_row.append(0)
            # idx
            new_row.append(p)
            for neg in sampled:
                new_row.append(neg)
            new_row.append(uid)
            
            data_list.append(new_row)
    
    datanp = np.array(data_list)
    np.save(fout, datanp)

def main(args):
    f_train_beh = os.path.join(args.root, args.fsamples)
    if args.dataset == 'MIND':
        df = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
    else:
        df = pd.read_csv(f_train_beh, sep="\t", encoding="utf-8", names=["id", "uid", "imp"])
    news_info = json.load(open('{}/news.json'.format(args.root), 'r', encoding='utf-8'))
    user_info = json.load(open('{}/user.json'.format(args.root), 'r', encoding='utf-8'))

    subdf_len = math.ceil(len(df) / args.processes)
    cut_indices = [x * subdf_len for x in range(1, args.processes)]
    dfs = np.split(df, cut_indices)

    processes = []
    for i in range(args.processes):
        output_path = os.path.join(args.root, args.fout,  "train-{}.npy".format(i))
        p = mp.Process(target=build_examples, args=(
            i, args, dfs[i], news_info, user_info, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    print('concate')
    # concate
    data_list = []
    for i in range(args.processes):
        output_path = os.path.join(args.root, args.fout,  "train-{}.npy".format(i))
        data_list.append(np.load(output_path))
    datanp = np.concatenate(data_list, axis=0)
    print(datanp.shape)
    # delete origin
    print('delete')
    for i in range(args.processes):
        output_path = os.path.join(args.root, args.fout,  "train-{}.npy".format(i))
        os.remove(output_path)
    print('save splited file')
    sub_len = math.floor(len(datanp) / 8)
    for i in range(8):
        output_path = os.path.join(args.root, args.fout,  "train-{}.npy".format(i))
        s = i * sub_len
        e = (i + 1) * sub_len
        np.save(output_path, datanp[s: e])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path options.
    parser.add_argument("--fsamples", default="train/target_behaviors.tsv", type=str,
                        help="Path of the training samples file.")
    parser.add_argument("--fout", default="raw", type=str,
                        help="Path of the output dir.")
    parser.add_argument("--neg_count", default=4, type=int)
    parser.add_argument("--processes", default=40, type=int, help="Processes number")
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--dataset", default="MIND", type=str)
    args = parser.parse_args()

    main(args)

