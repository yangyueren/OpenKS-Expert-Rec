
from collections import defaultdict
import os
from tqdm import tqdm
import torch
import torch.nn  as nn
import torch.nn.functional  as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from util import save_to_disk

from util import load_from_disk
from mylogger import logger

class TwoTowerModel(nn.Module):
    def __init__(self, config):
        super(TwoTowerModel, self).__init__()
        self.config = config
        person = load_from_disk(config.entities_person)
        logger.info(f'person num is {len(person)}')
        person = sorted(person)

        self.person_emb = self.get_person_emb(config, person)
        # import pdb; pdb.set_trace()
        

        self.fc1 = nn.Linear(768*2, 768)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(768, 1)
        self.bert = AutoModel.from_pretrained(config.bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    def get_person_emb(self, config, person):
        if config.debug:
            return nn.Embedding(len(person), 768)
        else:
            if os.path.exists('./log/person_emb.pkl'):
                person_emb = load_from_disk('./log/person_emb.pkl')
                assert len(person) == len(person_emb)
                return nn.Embedding.from_pretrained(person_emb, freeze=False)
            
            paper_emb = load_from_disk(config.paper_emb)
            rel_is_publisher_of = load_from_disk(config.rel_is_publisher_of)
            from collections import defaultdict
            person2paper = defaultdict(set)
            for triple in rel_is_publisher_of:
                _, personid, paperid, _ = triple
                person2paper[personid].add(paperid)
            paper2emb = defaultdict(list)
            for triple in paper_emb:
                paperid, title_emb, abstract_emb = triple
                emb = (torch.tensor(title_emb) + torch.tensor(abstract_emb)) / 2
                paper2emb[paperid].append(emb)
            person_emb = []

            for per in tqdm(sorted(person)):
                emb = []
                if per in person2paper:
                    for paper in person2paper[per]:
                        if paper in paper2emb:
                            emb.append(torch.tensor(paper2emb[paper][0]))
                if len(emb) == 0:
                    emb.append(torch.randn(768))
                
                emb = torch.vstack(emb)
                emb = emb.mean(dim=0)
                person_emb.append(emb)
            person_emb = torch.vstack(person_emb)
            
            assert len(person) == len(person_emb)
            save_to_disk(person_emb, './log/person_emb.pkl')
            

            return nn.Embedding.from_pretrained(person_emb, freeze=False)


    def forward(self, person_ids, text):
        
        person = self.person_emb(person_ids)
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs.to(self.config.device)
        outs = self.bert(output_attentions=True, **inputs)
        text_emb = outs.last_hidden_state[:,0,:]

        mix = torch.cat([person, text_emb.unsqueeze(1).repeat(1, person.shape[1], 1)], dim=-1)
        mix = self.fc1(mix)
        mix = F.relu(self.drop(mix))

        logits = self.fc2(mix).squeeze()
        return logits
