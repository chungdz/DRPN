import os
import argparse
import json
import pickle
from tqdm import tqdm
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import logging
from torch_geometric.data import DataLoader
import math
from datasets.recodata import RecoData
from datasets.config import ModelConfig
from gather import gather as gather_all
from models.pnrec import PNRec
from utils.log_util import convert_omegaconf_to_dict
from utils.train_util import set_seed
from utils.train_util import save_checkpoint_by_epoch
from utils.eval_util import group_labels
from utils.eval_util import cal_metric

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run(cfg, rank, dev_dataset, device, model):
    set_seed(7)

    model.to(device)
    model.eval()

    valid_data_loader = DataLoader(dev_dataset, batch_size=cfg.batch_size, shuffle=False)

    if ((cfg.gpus < 2) or (cfg.gpus > 1 and rank == 0)):
        data_iter = tqdm(enumerate(valid_data_loader),
                        desc="EP_dev:%d" % 1,
                        total=len(valid_data_loader),
                        bar_format="{l_bar}{r_bar}")
    else:
        data_iter = enumerate(valid_data_loader)

    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:

            imp_ids += data.imp_id.cpu().numpy().tolist()
            truths += data.y.cpu().long().cpu().numpy().tolist()
            data = data.to(device)

            # 1. Forward
            pred = model(data, test_mode=True)
            if pred.dim() > 1:
                pred = pred.squeeze()
            try:
                preds += pred.cpu().numpy().tolist()
            except:
                print(data.size())
                preds.append(int(pred.cpu().numpy()))

        tmp_dict = {}
        tmp_dict['imp'] = imp_ids
        tmp_dict['labels'] = truths
        tmp_dict['preds'] = preds

        with open(cfg.result_path + 'tmp_small_{}.json'.format(rank), 'w', encoding='utf-8') as f:
            json.dump(tmp_dict, f)

def gather(cfg, turn, validate=False):
    output_path = cfg.result_path
    filenum = cfg.gpus

    preds = []
    labels = []
    imp_indexes = []

    for i in range(filenum):
        with open(output_path + 'tmp_small_{}.json'.format(i), 'r', encoding='utf-8') as f:
            cur_result = json.load(f)
        imp_indexes += cur_result['imp']
        labels += cur_result['labels']
        preds += cur_result['preds']

    tmp_dict = {}
    tmp_dict['imp'] = imp_indexes
    tmp_dict['labels'] = labels
    tmp_dict['preds'] = preds

    with open(cfg.result_path + 'tmp_{}.json'.format(turn), 'w', encoding='utf-8') as f:
        json.dump(tmp_dict, f)


def split_dataset(dataset, gpu_count):
    sub_len = math.ceil(len(dataset) / gpu_count)
    data_list = []
    for i in range(gpu_count):
        s = i * sub_len
        e = (i + 1) * sub_len
        data_list.append(dataset[s: e])

    return data_list


def main(cfg):
    set_seed(7)

    file_num = cfg.filenum
    cfg.result_path = './result/'
    print('load config')
    model_cfg = ModelConfig(cfg.root)
    cfg.mc = model_cfg

    model = PNRec(model_cfg)

    saved_model_path = os.path.join('./checkpoint/', 'model.ep{0}'.format(cfg.epoch))
    print("Load from:", saved_model_path)
    if not os.path.exists(saved_model_path):
        print("Not Exist: {}".format(saved_model_path))
        return []
    model.cpu()
    pretrained_model = torch.load(saved_model_path, map_location='cpu')
    print(model.load_state_dict(pretrained_model, strict=False))

    for point_num in range(cfg.start_dev, file_num):
        print("processing {}/raw/dev-{}.pt".format(cfg.root, point_num))
        valid_dataset = RecoData(cfg.root, 'dev', point_num)
        dataset_list = split_dataset(valid_dataset, cfg.gpus)
        
        processes = []
        for rank in range(cfg.gpus):
            cur_device = torch.device("cuda:{}".format(rank))

            p = mp.Process(target=run, args=(cfg, rank, dataset_list[rank], cur_device, model))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        
        gather(cfg, point_num)
    
    gather_all(cfg.result_path, file_num, start_file=cfg.start_dev, validate=True, save=True)
        

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenum', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--gpus', type=int, default=2, help='gpu_num')
    parser.add_argument('--epoch', type=int, default=0, help='the number of epochs load checkpoint')
    parser.add_argument("--root", default="data", type=str)
    parser.add_argument("--start_dev", default=2, type=int)
    opt = parser.parse_args()
    logging.warning(opt)

    main(opt)
