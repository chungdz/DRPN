import os
import json
import pickle
import argparse
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from prepocess.embd import build_word_embeddings, build_news_embeddings
from prepocess.graph_utils import build_entire_graph

def parse_ent_list(x):
    if x.strip() == "":
        return ''

    return ' '.join([k["WikidataId"] for k in json.loads(x)])

punctuation = '!,;:?"\''
def removePunctuation(text):
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text.strip().lower()

parser = argparse.ArgumentParser()

parser.add_argument("--title_len", default=15, type=int, help="Max length of the title.")
parser.add_argument("--pos_hist_length", default=30, type=int)
parser.add_argument("--neg_hist_length", default=60, type=int)

args = parser.parse_args()

data_path = 'data'
max_title_len = args.title_len

print("Loading news info")
f_train_news = os.path.join(data_path, "train/news.tsv")
f_dev_news = os.path.join(data_path, "dev/news.tsv")

print("Loading training news")
all_news = pd.read_csv(f_train_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)

print("Loading dev news")
dev_news = pd.read_csv(f_dev_news, sep="\t", encoding="utf-8",
                        names=["newsid", "cate", "subcate", "title", "abs", "url", "title_ents", "abs_ents"],
                        quoting=3)
all_news = pd.concat([all_news, dev_news], ignore_index=True)

all_news = all_news.drop_duplicates("newsid")

news_dict = {}
word_dict = {'<pad>': 0}
word_idx = 1
news_idx = 2
for n, title, cate, subcate in tqdm(all_news[['newsid', "title", "cate", "subcate"]].values, total=all_news.shape[0], desc='parse news'):
    news_dict[n] = {}
    news_dict[n]['idx'] = news_idx
    news_idx += 1

    tarr = removePunctuation(title).split()
    tarr = [cate, subcate] + tarr
    wid_arr = []
    for t in tarr:
        if t not in word_dict:
            word_dict[t] = word_idx
            word_idx += 1
        wid_arr.append(word_dict[t])
    cur_len = len(wid_arr)
    if cur_len < max_title_len:
        for l in range(max_title_len - cur_len):
            wid_arr.append(0)
    news_dict[n]['title'] = wid_arr[:max_title_len]

## paddning news for impression
news_dict['<pad>']= {}
news_dict['<pad>']['idx'] = 0
tarr = removePunctuation("This is the title of the padding news").split()
tarr = ['<pad>', '<pad>'] + tarr
wid_arr = []
for t in tarr:
    if t not in word_dict:
        word_dict[t] = word_idx
        word_idx += 1
    wid_arr.append(word_dict[t])
cur_len = len(wid_arr)
if cur_len < max_title_len:
    for l in range(max_title_len - cur_len):
        wid_arr.append(0)
news_dict['<pad>']['title'] = wid_arr[:max_title_len]

## paddning news for history
news_dict['<his>']= {}
news_dict['<his>']['idx'] = 1
news_dict['<his>']['title'] = list(np.zeros(max_title_len))

print('all word', len(word_dict))
print('all news', len(news_dict))


print("Loading behaviors info")
f_his_beh = os.path.join(data_path, "train/his_behaviors.tsv")
his_beh = pd.read_csv(f_his_beh, sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])

user_dict = {}
user_idx = 0
for uid, imp in tqdm(his_beh[['uid', 'imp']].values, total=his_beh.shape[0], desc='history behavior'):

    if uid not in user_dict:
        user_dict[uid] = {"pos": [], "neg": [], "idx": user_idx}
        user_idx += 1
    
    imp_list = str(imp).split(' ')
    for impre in imp_list:
        arr = impre.split('-')
        curn = news_dict[arr[0]]['idx']
        label = int(arr[1])
        if label == 0:
            user_dict[uid]["neg"].append(curn)
        elif label == 1:
            user_dict[uid]["pos"].append(curn)
        else:
            raise Exception('label error!')

dg = build_entire_graph([v['idx'] for v in news_dict.values()], user_dict)

for uid, uinfo in tqdm(user_dict.items(), total=len(user_dict), desc='pad history'):

    pos_len = len(uinfo["pos"])
    if pos_len < args.pos_hist_length:
        for _ in range(args.pos_hist_length - pos_len):
            uinfo["pos"].insert(0, news_dict['<his>']['idx'])
    else:
        uinfo["pos"] = uinfo["pos"][-args.pos_hist_length:]
    
    neg_len = len(uinfo["neg"])
    if neg_len < args.neg_hist_length:
        for _ in range(args.neg_hist_length - neg_len):
            uinfo["neg"].insert(0, news_dict['<his>']['idx'])
    else:
        uinfo["neg"] = uinfo["neg"][-args.neg_hist_length:]


build_word_embeddings(word_dict, 'data/glove.840B.300d.txt', 'data/emb.npy')
build_news_embeddings(news_dict, 'data/news_info.npy')
pickle.dump(dg, open('data/graph.bin', 'wb'))
json.dump(user_dict, open('data/user.json', 'w', encoding='utf-8'))
json.dump(news_dict, open('data/news.json', 'w', encoding='utf-8'))
json.dump(word_dict, open('data/word.json', 'w', encoding='utf-8'))

