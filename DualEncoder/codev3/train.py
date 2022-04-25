
import argparse
import os
import yaml
import shutil
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
from nsf_dataset import NSFDataset
from nsf_model import TwoTowerModel
from util import Dict2Obj, collect_fn

from mylogger import logger

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    

    test(config, model, test_dataset)
    logger.info(model)
    
    g_recall = 0

    for epoch in range(config.epochs):
        logger.info(f'epoch {epoch} begin...')
        model.train()

        for person, text, label in tqdm(train_dataloader):
            optimizer.zero_grad()
            
            person = person.to(config.device)
            label = label.to(config.device).float()

            logits = model(person, text) # logits
            # import pdb;pdb.set_trace()
            pos = logits[:, 0].squeeze()
            neg = logits[:, 1:]

            
            pos_loss = F.binary_cross_entropy_with_logits(pos, label)

            neg = neg.mean(dim=-1).squeeze()
            neg_loss = F.binary_cross_entropy_with_logits(neg, 1-label)

            loss = pos_loss + neg_loss

            
            loss.backward()
            optimizer.step()
            scheduler.step()
            preds = (pos >= 0.5).tolist() + (neg >= 0.5).tolist()
            label = label.tolist() + (1-label).tolist()
            prec, recall, f1 = train_evaluate(preds, label)

            logger.debug(f'epoch {epoch}, prec {prec:0.4f}, recall {recall:0.4f}, f1 {f1:0.4f} , loss {loss.cpu().data}' )
        if epoch > 0 and epoch % config.test_every_n_epoch == 0:
            recall = test(config, model, test_dataset)
            if recall > g_recall:
                g_recall = recall
                logger.info(f'save model with recall {recall.cpu().data}')
                torch.save(model.cpu().state_dict(), config.log_dir + './model.pth')
                model.to(config.device)

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

            logit = model(person, text) # logits
            
            rank = torch.argsort(logit, dim=1, descending=True)
            ranks.append(rank)
            
    ranks = torch.vstack(ranks)
    recall = evaluate(ranks, config.recall_k)
    recall1 = evaluate(ranks, 1)
    recall5 = evaluate(ranks, 5)
    recall20 = evaluate(ranks, 20)

    logger.info(f'test: num of data is {len(test_dataset)}, hit@1 {recall1:0.4f}, hit@5 {recall5:0.4f},  recall@ten  {recall:0.4f}, hit@20 {recall20:0.4f}')
    return recall


def copy_files(config):
    files = os.listdir('.')
    for file in files:
        if file.endswith('.py') or file.endswith('.yml'):
            shutil.copy(file, config.log_dir)


def main():
    
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--config_path', type=str, default='./config.yml',
                        help='config path')

    args = parser.parse_args()
    logger.info(args)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
        config = Dict2Obj(config).config
    logger.info(config)
    setup_seed(config.seed)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    copy_files(config)

    config.device = torch.device(f'cuda:{config.gpu}' if config.gpu >= 0 else 'cpu')
    logger.info(f'device {config.device}')
    train(config)
    
    
if __name__ == '__main__':
    main()