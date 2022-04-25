import torch
import torch.nn  as nn
import torch.nn.functional  as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from util import load_from_disk

class TwoTowerModel(nn.Module):
    def __init__(self, config):
        super(TwoTowerModel, self).__init__()
        self.config = config
        person = load_from_disk(config.entities_person)
        self.person_emb = nn.Embedding(len(person), 768)
        self.fc1 = nn.Linear(768*2, 768)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(768, 1)
        self.bert = AutoModel.from_pretrained(config.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    def forward(self, person_ids, text):
        
        person = self.person_emb(person_ids)
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs.to(self.config.device)
        outs = self.bert(output_attentions=True, **inputs)
        text_emb = outs.last_hidden_state[:,0,:]

        mix = torch.cat([person, text_emb], dim=-1)
        mix = self.fc1(mix)
        mix = F.relu(self.drop(mix))

        logits = self.fc2(mix).squeeze()
        return logits



    def predict(self, person_ids, text):
        
        person = self.person_emb(person_ids)
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs.to(self.config.device)
        outs = self.bert(output_attentions=True, **inputs)
        text_emb = outs.last_hidden_state[:,0,:]

        mix = torch.cat([person, text_emb.unsqueeze(1).repeat(1, person.shape[1], 1)], dim=-1)
        mix = self.fc1(mix)
        mix = F.relu(self.drop(mix))

        mix = self.fc2(mix).squeeze()
        logits = F.softmax(mix, dim=-1)
        return logits
