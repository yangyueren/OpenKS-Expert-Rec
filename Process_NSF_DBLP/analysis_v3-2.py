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
    print(f'saving {file_name}')
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
def load_from_disk(file_name):
    print(f'loading {file_name}')
    with open(file_name, 'rb') as f:
        obj = pkl.load(f)
    return obj

def get_orderset():
    return OrderedSet

# %% [markdown]
# ## 统计dblp和NSF中的author信息

# %% [markdown]
# ### 获取nsf的author信息

# %%

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

# %%

## part2 
print('This is part2, we will load nsf-data.pkl, nsf_authors.pkl, dblpv13.pkl')
with open('./raw/nsf-data.pkl', 'rb') as f:
    nsf_data = pkl.load(f)
with open('./data/nsf_authors.pkl', 'rb') as f:
    nsf_authors = pkl.load(f)
with open('./raw/dblpv13.pkl', 'rb') as f:
    papers = pkl.load(f)


dblp_authors = OrderedDict()
for paper in tqdm(papers):
    try:
        for au in paper['authors']:
            did = au['_id']
            if did not in dblp_authors:
                dblp_authors[did] = []
            dblp_authors[did].append(au['name'])
    except Exception as e:
        pass

print(f'total dblp authors {len(dblp_authors)}')

# %% [markdown]
# #### 保存dblp author pkl
# 格式
# ```sh
# dblpid: [name]
# ```

# %%
save_to_disk(dblp_authors, './data/dblp_authors.pkl')

# %%
with open('./data/dblp_authors.pkl', 'rb') as f:
    dblp_authors = pkl.load(f)


# %% [markdown]
# ## 为nsf中的人名匹配到dblp

# %% [markdown]
# ### 加载之前的数据

# %%
with open('./data/nsf_authors.pkl', 'rb') as f:
    nsf_authors = pkl.load(f)

with open('./data/dblp_authors.pkl', 'rb') as f:
    dblp_authors = pkl.load(f)

# %% [markdown]
# ### 进行匹配

# %%

# dblp: dblpid; [name]
# nsf: nsfid: [(NSF_ID, PI_FULL_NAME, email)]
from collections import defaultdict

nsf_name2id = OrderedDict()
for id in sorted(nsf_authors.keys(), key=str):
    for t in nsf_authors[id]:
        name = t[1]
        if name not in nsf_name2id:
            nsf_name2id[name] = OrderedSet()
        nsf_name2id[name].add(id)

sample_names = list(sorted(nsf_name2id.keys()))


# dblp_authors
# 
dblp_name2id = OrderedDict()
for id in sorted(dblp_authors.keys(), key=str):
    for name in dblp_authors[id]:
        if name not in dblp_name2id:
            dblp_name2id[name] = OrderedSet()
        c=dblp_name2id[name].add(id)
pop_names = list(sorted(dblp_name2id.keys()))

save_to_disk(nsf_name2id, './data/nsf_name2id.pkl')
save_to_disk(dblp_name2id, './data/dblp_name2id.pkl')

# %% [markdown]
# ### 执行namematcher脚本
# 为没一个nsf的author name匹配到dblp中的一个author

# %%

if first_run:
    from namematcher import NameMatcher
    name_matcher = NameMatcher()
    name_matcher.params['disc_initial'] = 0.9

    try:
        matches = name_matcher.find_closest_names(sample_names, pop_names)
    except Exception as e:
        import pdb; pdb.set_trace()
        print(e)

    save_to_disk(matches, './data/matches.pkl')

    for i in range(len(matches)):
        orig_name = sample_names[i]
        pop_name, pop_index, score = matches[i]
        print('For name: %s, best match: %s, score %f' % (orig_name, pop_name, score))
        if i > 50:
            break


# %% [markdown]
# ### 统计信息

# %%
print(f'nsf: author id num {len(nsf_authors)}, author name num {len(nsf_name2id)}')
print(f'dblp: author id num {len(dblp_authors)}, author name num {len(dblp_name2id)}')

# %% [markdown]
# ### 加载出matches.pkl
# ```json
# pop_name, pop_index, score = matches[i]
# 
# nsf_authors: nsfid: [(NSF_ID, PI_FULL_NAME, email)]
# dblp_authors: dblpid: [name]
# 
# 
# nsf_name2id : name: set(nsfid)
# dblp_name2id : name: set(dblpid)
# 
# 
# ```
# 
# 
# 从中挑选出match的人员
# ```json
# sample_nsf_auhtors : nsfid: [(NSF_ID, PI_FULL_NAME, email)]
# sample_dblp_authors : dblpid: [name]
# ```

# %%

with open('./data/matches.pkl', 'rb') as f:
    matches = pkl.load(f)

cnt = 0
chosen_nsf_names = []
chosen_dblp_names = []
for i in range(len(matches)):
	orig_name = sample_names[i]
	pop_name, pop_index, score = matches[i]
	if score >= 0.98:
		cnt += 1
		chosen_nsf_names.append(orig_name)
		chosen_dblp_names.append(pop_name)
		assert orig_name in nsf_name2id, 'error'

print(cnt, len(nsf_name2id), cnt/len(nsf_name2id))
	# print('For name: %s, best match: %s, score %f' % (orig_name, pop_name, score))
	# if i > 100:
	# 	break

sample_nsf_auhtors = OrderedDict()
sample_dblp_authors = OrderedDict()
sample_nsf2dblp = OrderedDict()

assert len(chosen_nsf_names) == len(chosen_dblp_names)
for nsf_name, dblp_name in zip(chosen_nsf_names, chosen_dblp_names):
    nsfids = nsf_name2id[nsf_name]
    dblpids = dblp_name2id[dblp_name]
    for nsfid in nsfids:
        for dblpid in dblpids:
            if nsfid not in sample_nsf2dblp:
                sample_nsf2dblp[nsfid] = OrderedSet()
            sample_nsf2dblp[nsfid].add(dblpid)

for nsf_name in chosen_nsf_names:
	nsfids = nsf_name2id[nsf_name]
	for id in nsfids:
		sample_nsf_auhtors[id] = nsf_authors[id]

for dblp_name in chosen_dblp_names:
	dblpids = dblp_name2id[dblp_name]
	for id in dblpids:
		sample_dblp_authors[id] = dblp_authors[id]

save_to_disk(sample_nsf_auhtors, './data/sample_nsf_authors.pkl')
save_to_disk(sample_dblp_authors, './data/sample_dblp_authors.pkl')


# %% [markdown]
# ## 挑选出与dblp相交的项目

# %% [markdown]
# ### 记录nsfid到dblpid的映射关系

# %%
# 相交的人员
sample_nsf_auhtors = load_from_disk('./data/sample_nsf_authors.pkl')
sample_dblp_authors = load_from_disk('./data/sample_dblp_authors.pkl')
print(len(sample_nsf_auhtors))
sample_nsf_authors_set = OrderedSet()
with open('./data/sample_nsf_authors.txt', 'w') as f:
    for k, v in sample_nsf_auhtors.items():
        f.write(f'{k},\t')
        for au in v:
            sample_nsf_authors_set.add(au)
            f.write(f'{au[0]}, {au[1]}, {au[2]},   \t')
        f.write(f'\n')

with open('./data/sample_nsf2dblp.txt', 'w') as f:
    for k in sorted(sample_nsf2dblp.keys()):
        for v in sample_nsf2dblp[k]:
            f.write(f'{k},{v}\n')
save_to_disk(sample_nsf2dblp, './data/sample_nsf2dblp.pkl')

# %% [markdown]
# ### 挑选出nsf的项目

# %%

sample_nsf_data = []
cnt = 0
for my_xml in tqdm(nsf_data):
    try:
        my_dict = xmltodict.parse(my_xml)
        invs = my_dict['rootTag']['Award']['Investigator']
        if not isinstance(invs, list):
            invs = [invs]
        flag = True
        for inv in invs:
            author = extract(inv)
            # print(author)
            if author not in sample_nsf_authors_set:
                flag = False
        if flag:
            sample_nsf_data.append(my_xml)
            if len(invs) > 1:
                cnt += 1
        # break
    except Exception as e:
        # print(e)
        # break
        pass
print(f'sample project num is {len(sample_nsf_data)}, author>1: {cnt}')

save_to_disk(sample_nsf_data, './data/sample_nsf_data.pkl')

# %% [markdown]
# ### 挑选出的dblp的数据

# %%
sample_dblp_author_ids = OrderedSet()
for k in sample_nsf2dblp.keys():
    sample_dblp_author_ids |= sample_nsf2dblp[k]

sample_dblp_paper_ids = OrderedSet()
for paper in papers:
    try:
        for au in paper['authors']:
            did = au['_id']
            if did in sample_dblp_author_ids:
                sample_dblp_paper_ids.add(paper['_id'])
        
        for ref in paper['references']:
            sample_dblp_paper_ids.add(ref)
    except Exception as e:
        pass

save_to_disk(sample_dblp_author_ids, './data/sample_dblp_author_ids.pkl')
save_to_disk(sample_dblp_paper_ids, './data/sample_dblp_paper_ids.pkl')

# with open('./data/sample_dblp_author_ids.txt', 'w') as f:
#     for idx, aid in enumerate(sample_dblp_author_ids):
#         f.write(f'{aid}\n')

# with open('./data/sample_dblp_paper_ids.txt', 'w') as f:
#     for idx, pid in enumerate(sample_dblp_paper_ids):
#         f.write(f'{pid}\n')
print(f'found {len(sample_dblp_author_ids)} dblp authors, {len(sample_dblp_paper_ids)} dblp papers')

# %%
len(papers)

# %% [markdown]
# ## 对齐dblp中的人名

# %%
def rename_dblp(sample_nsf2dblp, sample_dblp_paper_ids, papers):
    dblpuid2nsfuid = OrderedDict()
    for k,v in sample_nsf2dblp.items():
        for vv in v:
            dblpuid2nsfuid[vv] = k

    sample_dblp_paper = []
    for paper in papers:
        if paper['_id'] in sample_dblp_paper_ids:
            p = copy.deepcopy(paper)
            rename_authors = []
            for au in p['authors']:
                try:
                    if au['_id'] in dblpuid2nsfuid:
                        au['_id'] = dblpuid2nsfuid[au['_id']]
                        rename_authors.append(au)
                except Exception as e:
                    pass
                
            p['authors'] = rename_authors
                # print(paper)
                # print(p)
            sample_dblp_paper.append(p)
    print(f'rename the author id in dblp: sample_dblp_paper: {len(sample_dblp_paper)}')
    return sample_dblp_paper

sample_dblp_paper = rename_dblp(sample_nsf2dblp, sample_dblp_paper_ids, papers)
save_to_disk(sample_dblp_paper, './data/sample_dblp_paper.pkl')

print('part2 end!!!')
