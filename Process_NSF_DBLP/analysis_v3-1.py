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
from yaml import load

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

# %% [markdown]
# ## 转换nsf数据到pickle

# %%
if first_run:
    nsf_data = []
    for sub in tqdm(sorted(subfolder)):
        if not os.path.isdir(root_path+sub):
            continue
        xmls = os.listdir(root_path+sub)
        # print(sub, len(xmls))
        for xml in sorted(xmls):
            if xml.endswith('.xml'):
                path = os.path.join(root_path+sub, xml)
                with open(path, 'r') as f:
                    content = f.read()
                    nsf_data.append(content)
    print('saving ./raw/nsf-data.pkl ...')
    save_to_disk(nsf_data, './raw/nsf-data.pkl')

# %%
with open('./raw/nsf-data.pkl', 'rb') as f:
    nsf_data = pkl.load(f)

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

def add_to_nsf_authors(author, nsf_authors_re):
    nsf_authors_re.add(author)


nsf_authors_re = OrderedSet()

for my_xml in nsf_data:
    try:
        my_dict = xmltodict.parse(my_xml)
        invs = my_dict['rootTag']['Award']['Investigator']
        if not isinstance(invs, list):
            invs = [invs]

        for inv in invs:
            author = extract(inv)
            add_to_nsf_authors(author, nsf_authors_re)

    except Exception as e:
        pass


# %%

def filter(nsf_authors_re):
    base = 0
    id2authors = OrderedDict()
    email2id = OrderedDict()
    for au in sorted(nsf_authors_re, key=str):
        def check(au):
            # nsfid
            if au[0] is not None and au[0] in id2authors:
                return au[0]
            # email
            if au[2] is not None and au[2] in email2id:
                return email2id[au[2]]
            return None

        uid = check(au)
        if uid is None:
            myuid = f'nsfuid{base:06d}'
            base += 1
            id2authors[myuid] = [au]
            if au[2] is not None:
                email2id[au[2]] = myuid
        else:
            id2authors[uid].append(au)
            if au[2] is not None:
                email2id[au[2]] = uid
    return id2authors


'''nsf_authors的格式
nsfuid000156 [(None, 'Jeffrey Draper', 'jberger@resodyn.com'), (None, 'Richard Williams', 'jberger@resodyn.com'), (None, 'Manfred Biermann', 'jberger@resodyn.com'), (None, 'Peter Lucon', 'jberger@resodyn.com')]
nsfuid000316 [('000275905', 'Ram Tenkasi', 'rtenkasi@ben.edu'), ('000531875', 'Ramkrishnan V Tenkasi', 'rtenkasi@ben.edu'), (None, 'Ram Tenkasi', 'rtenkasi@ben.edu')]
nsfuid001138 [('000450899', 'Anneke M Metz', 'anneke.metz@gmail.com'), ('000623195', 'Anneke Metz', 'anneke.metz@gmail.com'), (None, 'Anneke Metz', 'anneke.metz@gmail.com')]
nsfuid002136 [(None, 'Toufic Hakim', 'thakim@aapt.org'), ('000238536', 'Toufic M Hakim', 'thakim@aapt.org'), ('000077199', 'Toufic Hakim', 'thakim@aapt.org')]
'''

nsf_authors = filter(nsf_authors_re)
print(f'nsf authors num: {len(nsf_authors)}')

# %% [markdown]
# #### 保存nsf author pkl
# nsf_authors的字典格式 nsfid: [(NSF_ID, PI_FULL_NAME, email)]
# ```sh
# nsfuid000156 : [(None, 'Jeffrey Draper', 'jberger@resodyn.com'), (None, 'Richard Williams', 'jberger@resodyn.com'), (None, 'Manfred Biermann', 'jberger@resodyn.com'), (None, 'Peter Lucon', 'jberger@resodyn.com')]
# nsfuid000316 : [('000275905', 'Ram Tenkasi', 'rtenkasi@ben.edu'), ('000531875', 'Ramkrishnan V Tenkasi', 'rtenkasi@ben.edu'), (None, 'Ram Tenkasi', 'rtenkasi@ben.edu')]
# nsfuid001138 : [('000450899', 'Anneke M Metz', 'anneke.metz@gmail.com'), ('000623195', 'Anneke Metz', 'anneke.metz@gmail.com'), (None, 'Anneke Metz', 'anneke.metz@gmail.com')]
# nsfuid002136 : [(None, 'Toufic Hakim', 'thakim@aapt.org'), ('000238536', 'Toufic M Hakim', 'thakim@aapt.org'), ('000077199', 'Toufic Hakim', 'thakim@aapt.org')]
# ```

# %%
save_to_disk(nsf_authors, './data/nsf_authors.pkl')


# %%
with open('./data/nsf_authors.pkl', 'rb') as f:
    nsf_authors = pkl.load(f)

# %% [markdown]
# ### 获取所有的dblp的author信息

# %%
if first_run:
    import json as json
    import re
    dblp_path = './raw/dblpv13.json'
    out_path = './raw/dblpv13-j.json'
    # dblp_path = './a.json'
    # out_path = './aa.json'

    def func(matched):
        # print(matched.group())
        s = matched.group()[10:-1]
        return s

    with open(out_path, 'w') as g:
        with open(dblp_path, 'r') as f:
            for line in f.readlines():
                s = re.sub(r'NumberInt\(.*?\)', func, line)
                # print(s)
                g.write(s)

# %% [markdown]
# #### 保存dblp paper pkl

# %%
if first_run:
    with open(out_path, 'r') as f:
        papers = json.load(f)
    print('saving ./raw/dblpv13.pkl ...')
    save_to_disk(papers, './raw/dblpv13.pkl')

# %%
with open('./raw/dblpv13.pkl', 'rb') as f:
    papers = pkl.load(f)

"""
对已经生成的文件计算 md5sum （在shell中计算）
(base) zy@zju:~/data2/yyr/codes/kg4proj-rec$ md5sum ./raw/nsf-data.pkl 
ccbf680503438702a1b5047f0e4ec760  ./raw/nsf-data.pkl
(base) zy@zju:~/data2/yyr/codes/kg4proj-rec$ md5sum ./raw/dblpv13.pkl 
02aad7bfee401f412556ee134a11bda9  ./raw/dblpv13.pkl
(base) zy@zju:~/data2/yyr/codes/kg4proj-rec$ md5sum ./data/nsf_authors.pkl 
c773f451c99611db1de18035a3bbcebd  ./data/nsf_authors.pkl
(base) zy@zju:~/data2/yyr/codes/kg4proj-rec$ 
"""
