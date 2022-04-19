
import argparse

import yaml
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


from sklearn.metrics import precision_score, recall_score, f1_score

from nsf_dataset import NSFDataset
from nsf_model import TwoTowerModel
from util import Dict2Obj, collect_fn

from mylogger import logger

def train(config):
    train_dataset = NSFDataset(config, config.train_data)
    val_dataset = NSFDataset(config, config.val_data, is_train=False)
    test_dataset = NSFDataset(config, config.test_data, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collect_fn)

    model = TwoTowerModel(config)
    model.to(config.device)


    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    total_steps = int(len(train_dataloader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    

    test(config, model, test_dataset)

    for epoch in range(config.epochs):
        logger.info(f'epoch {epoch} begin...')
        model.train()

        for person, text, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            person = person.to(config.device)
            label = label.to(config.device).float()

            logits = model(person, text) # logits
            # import pdb;pdb.set_trace()
            preds = logits >= 0.5
            loss = F.binary_cross_entropy_with_logits(logits, label)
            
            loss.backward()
            optimizer.step()
            # scheduler.step()
            prec, recall, f1 = train_evaluate(preds.tolist(), label.tolist())
            logger.debug(f'epoch {epoch}, prec {prec:0.4f}, recall {recall:0.4f}, f1 {f1:0.4f} , loss {loss.cpu().data}' )
        if epoch > 0 and epoch % config.test_every_n_epoch == 0:
            test(config, model, test_dataset)

def train_evaluate(preds, labels):
    prec = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    return prec, recall, f1


def evaluate(rank, k=10):
    rank = rank[:, :k]
    recall = (rank == 0).sum()
    return recall / len(rank)



def test(config, model, test_dataset):
    logger.info('begin test...')
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=False, collate_fn=collect_fn)

    ranks = []
    model.eval()
    with torch.no_grad():
        for person, text, label in tqdm(test_dataloader):
            person = person.to(config.device)
            label = label.to(config.device)

            logit = model.predict(person, text) # logits
            
            rank = torch.argsort(logit, dim=1, descending=True)
            ranks.append(rank)
            
    ranks = torch.vstack(ranks)
    

    recall = evaluate(ranks, config.recall_k)

    logger.info(f'test: num of data is {len(test_dataset)},  recall@ten  {recall:0.4f}')



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--config_path', type=str, default='./config.yml',
                        help='config path')

    args = parser.parse_args()
    logger.info(args)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = Dict2Obj(config).config
    logger.info(config)
    torch.manual_seed(config.seed)
    config.device = torch.device(f'cuda:{config.gpu}' if config.gpu >= 0 else 'cpu')
    logger.info(f'device {config.device}')
    train(config)
    
    
    
if __name__ == '__main__':
    main()