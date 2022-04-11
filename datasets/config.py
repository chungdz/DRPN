import json
import pickle
import numpy as np

class ModelConfig():
    def __init__(self, root):

        # word_dict = json.load(open('data/word.json', 'r', encoding='utf-8'))
        news_dict = json.load(open('{}/news.json'.format(root), 'r', encoding='utf-8'))
        self.news_num = len(news_dict)
        self.word_emb = np.load('{}/emb.npy'.format(root))
        self.word_num = len(self.word_emb)

        self.pos_hist_length = 30
        self.neg_hist_length = 60
        self.neighbor_length = 90
        self.max_title_len = 15
        self.neg_count = 4
        self.word_dim = 300
        self.hidden_size = 300
        self.head_num = 6
        self.dropout = 0.2
        
        return None