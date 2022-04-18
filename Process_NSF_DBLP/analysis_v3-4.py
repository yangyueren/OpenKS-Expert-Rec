

import os
import re
import numpy as np
import pickle as pkl
import os
import re
import numpy as np
import pickle as pkl
import xmltodict
import pprint
from tqdm import tqdm
import copy
from collections import defaultdict, OrderedDict
import ujson as json
from ordered_set import OrderedSet

import random
random.seed(37)

def save_to_disk(obj, file_name):
    print(f'saving {file_name}')
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
def load_from_disk(file_name):
    print(f'loading {file_name}')
    with open(file_name, 'rb') as f:
        obj = pkl.load(f)
    return obj

entities_person = load_from_disk('./data/nsfkg/entities_person.pkl')

entities_project = load_from_disk('./data/nsfkg/entities_project.pkl')


def task1_dataset(entities_project, entities_person, year_fn):
    triples = []
    for proj in entities_project:
        p = json.loads(proj)
        Investigator = p['Investigator']
        AwardID = p['AwardID']
        AwardEffectiveDate = p['AwardEffectiveDate']
        year = int(AwardEffectiveDate[-4:])
        invs = OrderedSet()
        pricipal_uid = None
        for inv in Investigator:
            invs.add(inv['uid'])
            if inv['RoleCode'] == 'Principal Investigator':
                pricipal_uid = inv['uid']
        if year_fn(year):
            neg_persons = OrderedSet()
            while len(neg_persons) < 99:
                idx = random.randint(0,len(entities_person)-1)
                if entities_person[idx] not in invs:
                    neg_persons.add(entities_person[idx])
            if pricipal_uid is None or len(p['AbstractNarration']) < 10 or AwardID is None:
                continue
                
            triples.append( ('is_principal_investigator_of', pricipal_uid, AwardID, year,list(neg_persons) )) 
    print(f'is_principal_investigator_of {len(triples)}, each with 99 negative persons.')
    return triples

def train_year_fn(year):
    if year < 2015:
        return True
train_rel_is_principal_investigator_of = task1_dataset(entities_project, entities_person, train_year_fn)
print(train_rel_is_principal_investigator_of[0])


def test_year_fn(year):
    if year >= 2015:
        return True
is_principal_investigator_of = task1_dataset(entities_project, entities_person, test_year_fn)
val_rel_is_principal_investigator_of = []
test_rel_is_principal_investigator_of = []
for triple in is_principal_investigator_of:
    if random.random() < 0.3:
        val_rel_is_principal_investigator_of.append(triple)
    else:
        test_rel_is_principal_investigator_of.append(triple)
print(f'val_rel_is_principal_investigator_of {len(val_rel_is_principal_investigator_of)}')
print(f'test_rel_is_principal_investigator_of {len(test_rel_is_principal_investigator_of)}')
save_to_disk(train_rel_is_principal_investigator_of, './data/nsfkg/train_rel_is_principal_investigator_of.pkl')
save_to_disk(val_rel_is_principal_investigator_of, './data/nsfkg/val_rel_is_principal_investigator_of.pkl')
save_to_disk(test_rel_is_principal_investigator_of, './data/nsfkg/test_rel_is_principal_investigator_of.pkl')

