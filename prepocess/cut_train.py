import pandas as pd
import numpy as np
import json

def extract_time(x):

    clock = x.split()[1]
    arr = clock.split(':')
    int_clock = int(arr[0]) * 10000 + int(arr[1]) * 100 + int(arr[2])

    return int_clock

train = pd.read_csv('./data/train/behaviors.tsv', sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
train['date'] = train['time'].apply(lambda x: x.split()[0])
his_day = train[train['date'] != '11/14/2019']
target_day = train[train['date'] == '11/14/2019']

his_day.drop(columns=['date']).to_csv('./data/train/his_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('his ', his_day.shape)
target_day.drop(columns=['date']).to_csv('./data/train/target_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('target ', target_day.shape)

dev = pd.read_csv('./data/dev/behaviors.tsv', sep="\t", encoding="utf-8", names=["id", "uid", "time", "hist", "imp"])
dev['clock'] = dev['time'].apply(lambda x: extract_time(x))
dev.sort_values('clock', axis=0, inplace=True, ignore_index=True)
dev.drop(columns=['clock']).to_csv('./data/dev/target_behaviors.tsv', sep="\t", encoding="utf-8", header=None, index=None)
print('dev ', dev.shape)
