import torch
import time
import numpy as np
import dgl
from collections import defaultdict
from torch.utils.data import Dataset

class NSFDataset(Dataset):
    def __init__(self, data, args):
        super(NSFDataset, self).__init__()
        self.data = data

    def __getitem__(self, index):
        project_id = self.data[index][0]
        project_text_emb = self.data[index][1]
        pos_person = self.data[index][2]
        neg_person_list = self.data[index][3]
        return project_id, project_text_emb, pos_person, neg_person_list

    def __len__(self):
        return len(self.data)
        # return 100
