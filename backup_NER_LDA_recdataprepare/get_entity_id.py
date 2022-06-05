import os
import re
import copy
from tqdm import tqdm

import math
import sys

import pickle as pkl
import ujson as json
def save_to_disk(obj, file_name):
    print('saving ', file_name)
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
def load_from_disk(file_name):
    print('loading ', file_name)
    with open(file_name, 'rb') as f:
        obj = pkl.load(f)
    return obj


import spacy  # version 3.0.6'

# initialize language model
nlp = spacy.load("en_core_web_md")

# add pipeline (declared through entry_points in setup.py)
nlp.add_pipe("entityLinker", last=True)


def parse_itself(e):
    span = e.get_span()
    # import pdb; pdb.set_trace()
    text = span.text
    pos = (span.start_char, span.end_char)
    wikiid = e.get_id()
    desc = e.get_description()
    url = e.get_url()
    label = e.get_label()
    itself = {
        'text': text,
        'pos': pos,
        'wiki_id': wikiid,
        'desc': desc,
        'url': url,
        'wiki_label': label
    }
    return itself

def parse_parent(e):
    
    wikiid = e.get_id()
    desc = e.get_description()
    url = e.get_url()
    label = e.get_label()
    parent = {
        'wiki_id': wikiid,
        'desc': desc,
        'url': url,
        'wiki_label': label
    }
    return parent

def parse_entity_element(e):
    
    itself = parse_itself(e)

    # parents = []
    # for pa in e.get_super_entities():
    #     p = parse_parent(pa)
    #     parents.append(p)
    
    res = {
        'itself': itself,
        # 'parents': parents
    }
    return res


def extract_one_doc(docu):
    if docu is None or len(docu) == 0:
        rese = {
            'entity': [],
            'wiki': []
        }
        return rese
    docu = docu.replace('\t', ' ')
    docu = docu.replace('<br/>', ' ')
    docu = docu.replace('\n', ' ')
    docu = re.sub(' +', ' ', docu)
    # print(docu)
    doc = nlp(docu)

    entity = []
    for ent in doc.ents:
        pos = (ent.start_char, ent.end_char)
        entity.append((ent.text, ent.label_, pos))

    wiki = []
    
    for entity_element in doc._.linkedEntities:
        ans = parse_entity_element(entity_element)
        
        wiki.append(ans)
    
    res = {
        'entity': entity,
        'wiki': wiki
    }
    return res

# doc = 'Elon Musk was born in South Africa. Bill Gates and Steve Jobs come from the United States'
# ans = extract_one_doc(doc)



"""
project:{
    'title':{
        'entity':[(text, type, (start_char, end_char))],
        'wiki':[
            {
                'itself':{ // EntityElement
                    'text': 'jjj',
                    'pos': (2,5),
                    'wikiid': 'Q389',
                    'description': 'dxxx',
                    'url': 'xxx',
                    'label': 'wikilabel'
                }
                'parents':[ // list EntityElement

                ]
                
            }
        ]
    }
}
"""


def load_be(begin, end, path):
    if end == -1:
        projects = load_from_disk(path)[begin:]
    else:
        projects = load_from_disk(path)[begin:end]
    return projects

def get_projects(begin=0, end=-1, path='/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_project.pkl'):
    projects = load_be(begin, end, path)
    ans = []
    with open('project_err.log', 'w') as f:
        for p in tqdm(projects):
            try:
                p = json.loads(p)
                title = '' if 'AwardTitle' not in p else p['AwardTitle']
                abs = '' if 'AbstractNarration' not in p else p['AbstractNarration']
                id = p['AwardID']

                anst = extract_one_doc(title)
                ansa = extract_one_doc(abs)
                pro = {
                    'AwardID': id,
                    'AwardTitle': anst,
                    'AbstractNarration': ansa
                }
                ss = json.dumps(pro)
                ans.append(ss)
            except Exception as e:
                print(e)
                f.write(str(id))
                f.write('\n')
    save_to_disk(ans, f'./project2wiki_{begin}_{end}.pkl')
    print(f'{begin} to {end} success project.')
    if end == -1:
        print('this is the last chunk of project.')

def get_papers(begin=0, end=-1, path='/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_paper.pkl'):
    papers = load_be(begin, end, path)
    ans = []
    with open('paper_err.log', 'w') as f:
        for p in tqdm(papers):
            try:
                title = '' if 'title' not in p else p['title']
                abstract = '' if 'abstract' not in p else p['abstract']
                id = p['_id']
                anst = extract_one_doc(title)
                ansa = extract_one_doc(abstract)
                pap = {
                    '_id': id,
                    'title': anst,
                    'abstract': ansa
                }
                pap = json.dumps(pap)
                ans.append(pap)
            except Exception as e:
                print(e)
                f.write(str(id))
                f.write('\n')

    save_to_disk(ans, f'./paper2wiki_{begin}_{end}.pkl')
    print(f'{begin} to {end} success paper.')
    if end == -1:
        print('this is the last chunk of paper.')

get_projects()
get_papers()
# get_projects(0, 2000)
# get_papers(0, 2000)
