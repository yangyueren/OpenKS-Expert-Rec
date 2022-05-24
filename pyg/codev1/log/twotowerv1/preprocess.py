

import ujson as json
import yaml
import math
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from transformers import BertTokenizer, BertModel


from util import load_from_disk, save_to_disk, Dict2Obj
from mylogger import logger


from graph_util import sampler_of_graph
import random
random.seed(37)


def preprocess(config):

    person = load_from_disk(config.entities_person)
    person = sorted(person)
    person2id = {}
    for p in person:
        person2id[p] = len(person2id)
    assert len(person2id) == len(set(person2id.keys())), 'error'
    save_to_disk(person, './data/person.pkl')
    save_to_disk(person2id, './data/person2id.pkl')
    


    project = load_from_disk(config.entities_project)
    projects = [json.loads(p) for p in project]
    project = sorted(projects, key=lambda x: x['AwardID'])
    project2id = {}
    for p in project:
        project2id[p['AwardID']] = len(project2id)
    assert len(project2id) == len(set(project2id.keys())), 'error'
    save_to_disk(project, './data/project.pkl')
    save_to_disk(project2id, './data/project2id.pkl')
    


    paper = load_from_disk(config.entities_paper)
    paper = sorted(paper, key=lambda x: x['_id'])
    paper2id = {}
    for p in paper:
        paper2id[p['_id']] = len(paper2id)
    assert len(paper2id) == len(set(paper2id.keys())), 'error'
    save_to_disk(paper, './data/paper.pkl')
    save_to_disk(paper2id, './data/paper2id.pkl')
    


    rel_co_author = load_from_disk(config.rel_co_author)
    edge = []
    for tri in rel_co_author:
        _, person1, person2, year = tri
        edge.append((person2id[person1], person2id[person2]))
    save_to_disk(edge, './data/rel_co_author.pkl')

    rel_cooperate = load_from_disk(config.rel_cooperate)
    edge = []
    for tri in rel_cooperate:
        _, person1, person2, year = tri
        edge.append((person2id[person1], person2id[person2]))
    save_to_disk(edge, './data/rel_cooperate.pkl')


    rel_is_publisher_of = load_from_disk(config.rel_is_publisher_of)
    edge = []
    for tri in rel_is_publisher_of:
        _, personid, paperid, year = tri
        edge.append((paper2id[paperid], person2id[personid]))
    save_to_disk(edge, './data/rel_published_by.pkl')


    rel_reference = load_from_disk(config.rel_reference)
    edge = []
    for tri in rel_reference:
        _, paperid, refid, year = tri
        if refid not in paper2id or paperid not in paper2id:
            continue
        edge.append((paper2id[refid], paper2id[paperid]))
    # src -> dst
    save_to_disk(edge, './data/rel_reference.pkl')
    

    train_rel_pricipal_investigator = load_from_disk(config.train_data)
    edge = []
    for tri in train_rel_pricipal_investigator:
        _, personid, projectid, year, neg_personids = tri
        edge.append((project2id[projectid], person2id[personid]))
    save_to_disk(edge, './data/rel_pricipal_investigator_by.pkl')



    def convert(rel):
        train = []
        for tri in rel:
            rel, personid, projectid, year, neg_personids = tri
            neg = [person2id[p] for p in neg_personids]
            train.append((rel, project2id[projectid], person2id[personid], year, neg))
        return train


    test_rel_pricipal_investigator = load_from_disk(config.test_data)
    val_rel_pricipal_investigator = load_from_disk(config.val_data)

    train = convert(train_rel_pricipal_investigator)
    test = convert(test_rel_pricipal_investigator)
    val = convert(val_rel_pricipal_investigator)
    save_to_disk(train, './data/train_rel_pricipal_investigator.pkl')
    save_to_disk(test, './data/test_rel_pricipal_investigator.pkl')
    save_to_disk(val, './data/val_rel_pricipal_investigator.pkl')




def encoder(entries, model, tokenizer, config):
    emb_res = []
    batch_size = 64
    with torch.no_grad():
        for index in tqdm(range(math.ceil(len(entries)/batch_size))):
            ents = entries[index*batch_size: (index+1)*batch_size]
            inputs = tokenizer(ents, padding=True, truncation=True, max_length=512, return_tensors="pt").to(config.device)
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            emb_res.append(last_hidden_states[:,0,:].detach().cpu())
    emb_res = torch.cat(emb_res, dim=0)
    return emb_res


def get_embedding(config):
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    model = BertModel.from_pretrained(config.bert_path)
    model = model.to(config.device)
    model.eval()

    # project = load_from_disk(config.entities_project)
    # projects = [json.loads(p) for p in project]
    # project = sorted(projects, key=lambda x: x['AwardID'])
    # project2id = {}
    # for p in project:
    #     project2id[p['AwardID']] = len(project2id)
    # assert len(project2id) == len(set(project2id.keys())), 'error'
    # project_text = []
    # for p in project:
    #     project_text.append(p['AwardTitle'] + ". " + p['AbstractNarration'])
    # assert len(project_text) == len(project2id), 'error'
    
    # project_emb = encoder(project_text, model, tokenizer, config)
    # assert len(project_emb) == len(project2id), 'error'
    # save_to_disk(project_emb, './data/project_emb.pkl')




    paper = load_from_disk(config.entities_paper)
    paper = sorted(paper, key=lambda x: x['_id'])
    paper2id = {}
    for p in paper:
        paper2id[p['_id']] = len(paper2id)
    assert len(paper2id) == len(set(paper2id.keys())), 'error'
    paper_text = []
    for p in paper:
        title = p['title'] if 'title' in p else ""
        abstract = p['abstract'] if 'abstract' in p else ""
        paper_text.append(title + '. ' + abstract)
    assert len(paper_text) == len(paper2id), 'error'
    paper_emb = encoder(paper_text, model, tokenizer, config)
    assert len(paper_emb) == len(paper2id), 'error'
    
    save_to_disk(paper_emb, './data/paper_emb.pkl')




if __name__ == '__main__':
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
        config = Dict2Obj(config).config
    # preprocess(config)
    config.device = torch.device('cuda:0')
    get_embedding(config)
