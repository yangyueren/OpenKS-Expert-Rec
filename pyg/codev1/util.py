import pickle as pkl
import torch
from mylogger import logger

def save_to_disk(obj, file_name):
    logger.debug(f'saving {file_name}')
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
def load_from_disk(file_name):
    logger.debug(f'loading {file_name}')
    with open(file_name, 'rb') as f:
        obj = pkl.load(f)
    return obj


class Dict2Obj(dict):
    def __getattr__(self, key):
        value = self.get(key)
        return Dict2Obj(value) if isinstance(value,dict) else value
    def __setattr__(self, key, value):
        self[key] = value


def collect_fn(batch):
    persons = torch.tensor([f[0] for f in batch])
    persons = persons.reshape(-1)
    projects =  torch.tensor([f[1] for f in batch])
    labels = torch.tensor([f[2] for f in batch])
    
    return persons, projects, labels
