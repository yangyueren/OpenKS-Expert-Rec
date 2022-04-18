
import argparse
from asyncio.log import logger
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup


from nsf_dataset import NSFDataset
from nsf_model import TwoTowerModel
from util import Dict2Obj, collect_fn

from mylogger import logger

def train(config):
    train_dataset = NSFDataset(config, config.train_data)
    val_dataset = NSFDataset(config, config.val_data)
    test_dataset = NSFDataset(config, config.test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collect_fn)

    model = TwoTowerModel(config)
    model.to(config.device)


    optimizer = AdamW(model.parameters(), lr=config.lr, eps=config.adam_epsilon)
    total_steps = int(len(train_dataloader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    for epoch in range(config.epochs):
        model.train()

        for person, label, text in train_dataloader:
            person = person.to(config.device)
            label = label.to(config.device)

            

            outputs = model(person, text) # logits, argsort
            # import pdb;pdb.set_trace()
            loss = F.cross_entropy(outputs[0], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            recall = evaluate(outputs[1], label.tolist())
            logger.debug(f'epoch {epoch}, recall@ten  {recall:0.4f} , loss {loss.cpu().data}' )
        if epoch % 20 == 0:
            test(config, model, test_dataset)

def evaluate(preds, labels, k=10):
    recall = 0
    for label, pred in zip(labels, preds):
        if label in pred[:k]:
            recall += 1
    return recall / len(labels)



def test(config, model, test_dataset):
    test_dataloader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, drop_last=False, collate_fn=collect_fn)
    labels = []
    preds = []
    model.eval()
    with torch.no_grad():
        for person, label, text in test_dataloader:
            person = person.to(config.device)
            label = label.to(config.device)

            outputs = model(person, text) # logits, argsort
            labels += label.tolist()
            preds += outputs[1]

    recall = evaluate(preds, labels)

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