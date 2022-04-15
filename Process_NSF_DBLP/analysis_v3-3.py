# %% [markdown]
# # 融合nsf和dblp

# %%
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

# pip install ordered-set

root_path = './raw/NSF-data/'
subfolder = os.listdir(root_path)

if not os.path.exists('./data/nsfkg'):
    os.mkdir('./data/nsfkg')

first_run = True

print('If you run this script for the first time, please set variable first_run=True')
if first_run:
    print('Current first_run=True')
else:
    print('Current first_run=False')

# %% [markdown]
# ## utils

# %%
import pickle as pkl
def save_to_disk(obj, file_name):
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
def load_from_disk(file_name):
    with open(file_name, 'rb') as f:
        obj = pkl.load(f)
    return obj

def get_orderset():
    return OrderedSet


def extract(d):
    PI_FULL_NAME = None
    NSF_ID = None
    email = None
    if 'PI_FULL_NAME' in d:
        PI_FULL_NAME = d['PI_FULL_NAME']
    else:
        mid = ' '
        if 'PI_MID_INIT' in d and d['PI_MID_INIT'] is not None:
            mid = ' ' + d['PI_MID_INIT'] + ' '
        PI_FULL_NAME = d['FirstName'] + mid + d['LastName']
    
    if 'EmailAddress' in d:
        email = d['EmailAddress']
    
    if 'NSF_ID' in d:
        NSF_ID = d['NSF_ID']
    return (NSF_ID, PI_FULL_NAME, email)




# %% [markdown]
# # 构建知识图谱

# %% [markdown]
# ## 加载所需数据

# %%
sample_nsf_authors = load_from_disk('./data/sample_nsf_authors.pkl')
print(f'sample_nsf_authors {len(sample_nsf_authors)}')

sample_dblp_authors = load_from_disk('./data/sample_dblp_authors.pkl')
print(f'sample_dblp_authors {len(sample_dblp_authors)}')

sample_nsf_data = load_from_disk('./data/sample_nsf_data.pkl')
print(f'sample_nsf_data {len(sample_nsf_data)}')

sample_dblp_paper_ids = load_from_disk('./data/sample_dblp_paper_ids.pkl')
print(f'sample_dblp_paper_ids {len(sample_dblp_paper_ids)}')

sample_nsf2dblp = load_from_disk('./data/sample_nsf2dblp.pkl')
print(f'sample_nsf2dblp {len(sample_nsf2dblp)}')

sample_dblp_paper = load_from_disk('./data/sample_dblp_paper.pkl')
print(f'sample_dblp_paper {len(sample_dblp_paper)}')



# %% [markdown]
# ## 验证实体库质量

# %%
# 实体类型：nsfuid，project，dblpuid，paper

# nsfuid 是 sample_nsf_authors
# project 是 sample_nsf_data
# dblpuid 是 sample_dblp_authors
# paper 是 sample_dblp_paper_ids


# %%
# cnt = 0
# for k,v in sample_nsf2dblp.items():
#     if len(v) == 1:
#         cnt += 1
# print(cnt, len(sample_nsf2dblp), cnt/len(sample_nsf2dblp))

# %%
# dates = []

# for data in sample_nsf_data:
#     my_dict = xmltodict.parse(data)
#     date = my_dict['rootTag']['Award']['AwardEffectiveDate'][-4:]
#     date = int(date)

#     invs = my_dict['rootTag']['Award']['Investigator']
#     if not isinstance(invs, list):
#         invs = [invs]

#     dates.append((date, len(invs)))

#     # break
# from collections import Counter
# c = Counter(dates)
# print(c)

# %%
# cyear = defaultdict(int)
# ctwo = defaultdict(int)
# for k in sorted(c.keys()):
#     cyear[k[0]] += c[k]
#     if k[1] > 1:
#         ctwo[k[0]] += c[k]
# print(cyear)
# print(ctwo)
# print(sum( [cyear[i] for i in cyear if i < 2017 and i >=2015] ) / 107125)
## 两人以上的队伍
## 15 16 val 0.106，17 18 19 20 test 0.1755; total 27192
# 一人
## 15 16 val 0.1068，17 18 19 20 test 0.1904; total 107125

# %%
## 划分训练集

'''
以2015年为界限，>=2015年的为val和test
entities_person.txt : uid

entities_project.txt: AwardID, AwardTitle, AwardEffectiveDate, AbstractNarration, Investigator
entities_paper.txt: paper_id, title, abstraction, authors, year

<uid, cooperate, uid, year>
<uid, co_author, uid, year>
<uid, is_publisher_of, paper_id, year>
<paper_id, reference, paper_id, year>

train
<uid, is_investigator_of, AwardID>

test
<uid, is_investigator_of, AwardID>

'''

# %%


# %%
# 

entities_person = []
# uid
for k in sample_nsf_authors:
    entities_person.append(k)
print(f'entities_person {entities_person[0]}')
save_to_disk(entities_person, './data/nsfkg/entities_person.pkl')

entities_paper = sample_dblp_paper
print(f'entities_paper {entities_paper[0]}')
save_to_disk(entities_paper, './data/nsfkg/entities_paper.pkl')

# %%
# entities_project.txt: AwardID, AwardTitle, AwardEffectiveDate, AbstractNarration, Investigator
def f_entities_project(sample_nsf_data, sample_nsf_authors):
    author2id = OrderedDict()
    for k,v in sample_nsf_authors.items():
        for vv in v:
            author2id[vv] = k
    print(len(author2id))
    cnt = 0
    entities_project = []
    for data in tqdm(sample_nsf_data):

        my_dict = xmltodict.parse(data)

        AwardID = my_dict['rootTag']['Award']['AwardID']
        AwardTitle = my_dict['rootTag']['Award']['AwardTitle']
        AwardEffectiveDate = my_dict['rootTag']['Award']['AwardEffectiveDate']
        AbstractNarration = my_dict['rootTag']['Award']['AbstractNarration']
        if AwardTitle is None:
            AwardTitle = ''
        if AbstractNarration is None:
            AbstractNarration = ''
        
        Investigator = []
        
        invs = my_dict['rootTag']['Award']['Investigator']
        
        if not isinstance(invs, list):
            invs = [invs]
        for inv in invs:
            # print(inv)
            author = extract(inv)
            # print(author)
            if author in author2id:
                au = {'uid': author2id[author], 'RoleCode': inv['RoleCode']}
                Investigator.append(au)
                cnt += 1
        project = {
            'AwardID': AwardID,
            'AwardTitle': AwardTitle,
            'AwardEffectiveDate': AwardEffectiveDate,
            'AbstractNarration': AbstractNarration,
            'Investigator': Investigator
        }
        s = json.dumps(project)
        entities_project.append(s)
        # break
    print(f'entities_project num is {len(entities_project)}')
    print('author num ', cnt)
    return entities_project
entities_project = f_entities_project(sample_nsf_data, sample_nsf_authors)
print(f'entities_project {entities_project[0]}')
save_to_disk(entities_project, './data/nsfkg/entities_project.pkl')
        


# %%
# <uid, cooperate, uid, year>

def f_cooperate(entities_person, entities_project):

    triples = OrderedSet()
    cnt = 0
    for project in tqdm(entities_project):
        pro = json.loads(project)
        # print(pro)

        year = int(pro['AwardEffectiveDate'][-4:])
        # print(year)
        cnt += len(pro['Investigator'])
        for au1 in pro['Investigator']:
            for au2 in pro['Investigator']:
                if au1['uid'] != au2['uid']:
                    triples.add(('cooperate', au1['uid'], au2['uid'], year))
        # break  
    print(cnt)
    print(f'entities_person: {len(entities_person)}, entities_project: {len(entities_project)},  cooperate: {len(triples)}')
    return list(triples)

cooperate = f_cooperate(entities_person, entities_project)
print(f'cooperate {cooperate[0]}')
save_to_disk(cooperate, './data/nsfkg/rel_cooperate.pkl')

    

# %%
# <uid, co_author, uid, year>
def f_co_author(entities_paper):
    triples = OrderedSet()
    cnt = 0
    for paper in tqdm(entities_paper):
        try:
            year = paper['year']
            auset = OrderedSet()
            for au in paper['authors']:
                auset .add(au['_id'])
            for au1 in auset:
                for au2 in auset:
                    if au1 != au2:
                        triples.add(('co_author', au1, au2, year))
        except:
            cnt += 1
            pass
        
            # print(auset)
            # print(paper)
            # break
    print(f'wrong paper {cnt}')
    print(f'entities_paper {len(entities_paper)}, co_author: {len(triples)}')
    return list(triples)

co_author = f_co_author(entities_paper)
print(f'co_author {co_author[0]}')
save_to_disk(co_author, './data/nsfkg/rel_co_author.pkl')

# %%
# <uid, is_publisher_of, paper_id, year>
def f_is_publisher_of(entities_paper):
    triples = OrderedSet()
    for paper in tqdm(entities_paper):
        if 'year' not in paper:
            continue
        year = paper['year']
        for au in paper['authors']:
            auid = au['_id']
            triples.add(('is_publisher_of', auid, paper['_id'], year))
    print(f'entities_paper {len(entities_paper)}, is_publisher_of: {len(triples)}')
    return list(triples)

is_publisher_of = f_is_publisher_of(entities_paper)
print(f'is_publisher_of {is_publisher_of[0]}')
save_to_disk(is_publisher_of, './data/nsfkg/rel_is_publisher_of.pkl')

# %%
# <paper_id, reference, paper_id, year>
def f_reference(entities_paper):
    triples = OrderedSet()
    cnt = 0 
    for paper in tqdm(entities_paper):
        if 'year' not in paper:
            cnt += 1
            continue
        if 'references' not in paper:
            cnt += 1
            continue
        year = paper['year']
        for ref in paper['references']:
            triples.add(('reference', paper['_id'], ref, year))
    print(f'wrong paper {cnt}')
    print(f'entities_paper {len(entities_paper)}, reference: {len(triples)}')
    return list(triples)

reference = f_reference(entities_paper)
print(f'reference {reference[0]}')
save_to_disk(reference, './data/nsfkg/rel_reference.pkl')

# %%
# <uid, is_investigator_of, AwardID, year>
def f_is_principal_investigator_of(entities_project):
    triples = OrderedSet()
    for proj in tqdm(entities_project):
        p = json.loads(proj)
        AwardID = p['AwardID']
        AwardTitle = p['AwardTitle']
        AwardEffectiveDate = p['AwardEffectiveDate']
        AbstractNarration = p['AbstractNarration']
        Investigator = p['Investigator']
        # print(AwardEffectiveDate)
        # break
        year = int(AwardEffectiveDate[-4:])
        for au in Investigator:
            if au['RoleCode'] == 'Principal Investigator':
                triples.add( ('is_principal_investigator_of', au['uid'], AwardID, year) )
    print(f'entities_project {len(entities_project)}, is_principal_investigator_of: {len(triples)}')
    return list(triples)

is_principal_investigator_of = f_is_principal_investigator_of(entities_project)
print(f'is_principal_investigator_of {is_principal_investigator_of[0]}')
save_to_disk(is_principal_investigator_of, './data/nsfkg/rel_is_principal_investigator_of.pkl')


# <uid, is_investigator_of, AwardID, year>
def f_is_investigator_of(entities_project):
    triples = OrderedSet()
    for proj in tqdm(entities_project):
        p = json.loads(proj)
        AwardID = p['AwardID']
        AwardTitle = p['AwardTitle']
        AwardEffectiveDate = p['AwardEffectiveDate']
        AbstractNarration = p['AbstractNarration']
        Investigator = p['Investigator']
        # print(AwardEffectiveDate)
        # break
        year = int(AwardEffectiveDate[-4:])
        for au in Investigator:
            triples.add( ('is_investigator_of', au['uid'], AwardID, year) )
    print(f'entities_project {len(entities_project)}, is_investigator_of: {len(triples)}')
    return list(triples)

is_investigator_of = f_is_investigator_of(entities_project)
print(f'is_investigator_of {is_investigator_of[0]}')
save_to_disk(is_investigator_of, './data/nsfkg/rel_is_investigator_of.pkl')


# %%
print('end!!!')


