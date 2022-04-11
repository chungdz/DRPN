import pickle
import torch
from torch_geometric.data import InMemoryDataset, Data
from itertools import repeat

class RecoData(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, splits, transform=None, pre_transform=None):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert phrase in ['train', 'dev']
        self.phrase = phrase
        self.splits = splits
        super(RecoData, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
     
    @property
    def raw_file_names(self):
        return ["{}-{}.npy".format(self.phrase, i) for i in range(self.splits * 2, (self.splits + 1) * 2)]
    
    @property
    def processed_file_names(self):
        return [self.phrase + '-{}.pt'.format(self.splits)]
    
    def download(self):
        pass
    
    def convert_to_list(self, in_data, in_slices):

        def _convert_one(idx):
            data = in_data.__class__()
            if hasattr(in_data, '__num_nodes__'):
                data.num_nodes = in_data.__num_nodes__[idx]

            for key in in_data.keys:
                item, slices = in_data[key], in_slices[key]
                start, end = slices[idx].item(), slices[idx + 1].item()
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[in_data.__cat_dim__(key, item)] = slice(start, end)
                elif start + 1 == end:
                    s = slices[start]
                else:
                    s = slice(start, end)
                data[key] = item[s]
            return data
        def _len():
            for item in in_slices.values():
                return len(item) - 1
            return 0
        cnt = _len()
        return [_convert_one(i) for i in range(cnt)]

    def process(self):
        data_list = []

        for file_idx, raw_path in enumerate(self.raw_paths):
            print("Processing-{}".format(raw_path))
            data, slices = torch.load(raw_path)
            data_list += self.convert_to_list(data, slices)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    @staticmethod
    def trim_seq(seq, length=30):
        slen = len(seq)

        if slen >= length:
            return seq[-length:]
        else:
            new_seq = []
            for t in range(length - slen):
                new_seq.append(0)
            return new_seq + seq
        
