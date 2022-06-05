import os
import re
import copy
from tqdm import tqdm
import stanza
import math

import pickle as pkl
import ujson as json
def save_to_disk(obj, file_name):
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
def load_from_disk(file_name):
    with open(file_name, 'rb') as f:
        obj = pkl.load(f)
    return obj



def get_docs(path='/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_project.pkl'):
    projects = load_from_disk(path)
    ans = []
    ids = []
    for p in tqdm(projects):
        p = json.loads(p)
        ans.append(p['AwardTitle'] + '. ' + p['AbstractNarration'])
        ids.append(p['AwardID'])
    return ans, ids



def get_papers(path='/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_paper.pkl'):
    papers = load_from_disk(path)
    ans = []
    ids = []
    for p in tqdm(papers):
        # import pdb;pdb.set_trace()
        title = '' if 'title' not in p else p['title']
        abstract = '' if 'abstract' not in p else p['abstract']
        ans.append(title + '. ' + abstract)
        ids.append(p['_id'])
    return ans, ids


def extract_ner(contents, ids):
    import stanza
    nlp = stanza.Pipeline(lang="en", processors='tokenize,ner,lemma') # Initialize the default English pipeline

    ners = []
    batch_size = 256
    for index in tqdm(range(math.ceil(len(contents)/batch_size))):
        ents = contents[index*batch_size: (index+1)*batch_size]
        idbatch = ids[index*batch_size: (index+1)*batch_size]
        in_docs = [stanza.Document([], text=d) for d in ents]
        out_docs = nlp(in_docs)
        cur_ners = []
        for idx, o in enumerate(out_docs):
            res = o.to_dict()
            cur_ners.append((idbatch[idx], res))
        ners += cur_ners
    return ners
# award, awardids = get_docs()
# ans = extract_ner(award, awardids)
# print(ans[0])
# save_to_disk(ans, './project2entity.pkl')

paper, paperids = get_papers()
ans = extract_ner(paper, paperids)
print(ans[0])
save_to_disk(ans, './paper2entity.pkl')