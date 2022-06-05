#!/usr/bin/env python
# coding: utf-8

# This notebook gives examples about how to generate:
# 
# word_dict.pkl: convert the words in news titles into indexes.
# 
# word_dict_all.pkl: convert the words in news titles and abstracts into indexes.
# 
# embedding.npy: pretrained word embedding matrix of words in word_dict.pkl
# 
# embedding_all.npy: pretrained embedding matrix of words in word_dict_all.pkl
# 
# vert_dict.pkl: convert news verticals into indexes.
# 
# subvert_dict.pkl: convert news subverticals into indexes.
# 
# uid2index.pkl: convert user ids into indexes.
# 

# In[ ]:





# In[1]:


import os
import sys
import copy
import pickle as pkl
from tqdm import tqdm
import re
import numpy as np
from collections import Counter


# In[2]:


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

def word_tokenize(sent):
    """Tokenize a sententence

    Args:
        sent: the sentence need to be tokenized

    Returns:
        list: words in the sentence
    """

    # treat consecutive words or special punctuation as words
    sent = sent.replace('<br/>', ' ')
    sent = sent.replace('\n', ' ')
    sent = re.sub(' +', ' ', sent)
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []





# ## 生成word dictionary

# In[3]:


glove_path = '/data2/yyr/dataset/glove.840B.300d.txt'
word_embedding_dim = 300
paper_path = '/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_paper.pkl'
project_path = '/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_project.pkl'


# In[4]:


def glove_words(glove_path):
    words = set()
    with open(glove_path, "r") as f:
        for line in tqdm(f):  # noqa: E741 ambiguous variable name 'l'
            # return l
            l = line.split()  # noqa: E741 ambiguous variable name 'l'
            word = l[0]
            words.add(word)
    return words


# In[5]:


projects = load_from_disk(project_path)
papers = load_from_disk(paper_path)


# In[6]:


word_cnt = Counter()
word_cnt_all = Counter()
glove_w = glove_words(glove_path)

for p in projects:
    p = json.loads(p)
    title = p['AwardTitle'] if 'AwardTitle' in p else ""
    abstract = p['AbstractNarration'] if 'AbstractNarration' in p else ""
    t = word_tokenize(title)
    t = [i for i in t if i in glove_w]
    a = word_tokenize(abstract) + t
    a = [i for i in a if i in glove_w]

    word_cnt.update(t)
    word_cnt_all.update(a)
    


# In[ ]:


for p in papers:
    title = p['title'] if 'title' in p else ""
    abstract = p['abstract'] if 'abstract' in p else ""
    t = word_tokenize(title)
    t = [i for i in t if i in glove_w]
    a = word_tokenize(abstract) + t
    a = [i for i in a if i in glove_w]
    word_cnt.update(t)
    word_cnt_all.update(a)


# In[ ]:





# In[ ]:


word_dict = {k: v+1 for k, v in zip(word_cnt, range(len(word_cnt)))}
word_dict_all = {k: v+1 for k, v in zip(word_cnt_all, range(len(word_cnt_all)))}
save_to_disk(word_dict, './data/word_dict.pkl')
save_to_disk(word_dict_all, './data/word_dict_all.pkl')


# In[ ]:





# ## 生成category
# 
# vert_dict.pkl: convert news verticals into indexes.
# 
# subvert_dict.pkl: convert news subverticals into indexes.

# In[5]:


project2topic_path = '/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/LDA/project2topic.pkl'
paper2topic_path = '/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/LDA/paper2topic.pkl'


# In[7]:


topic2id = {'null': 0}
topicset = set()
project2topic = load_from_disk(project2topic_path)
for pt in project2topic:
    p, t = pt
    topicset.add('topic_project' + str(t))
paper2topic = load_from_disk(paper2topic_path)
for pt in paper2topic:
    p, t = pt
    topicset.add('topic_paper' + str(t))

for t in topicset:
    topic2id[t] = len(topic2id)

print(len(topic2id))
save_to_disk(topic2id, './data/vert_dict.pkl')
save_to_disk(topic2id, './data/subvert_dict.pkl')


# In[ ]:


# venue = set()
# cnt = 0
# cnt_raw = 0
# cnt_id = 0
# cnt_name_d = 0
# raw, id = set(), set()
# for p in papers:
#     paperid = p['_id']
#     if 'venue' not in p:
#         cnt+=1
#     else:
#         if 'raw' not in p['venue']:
#             cnt_raw += 1
#         else:
#             raw.add(p['venue']['raw'])

# for r in raw:
#     topic2id[r] = len(topic2id)



# In[ ]:





# ## 拿到embedding

# In[ ]:


def load_glove_matrix(glove_path, word_dict, word_embedding_dim):
    """Load pretrained embedding metrics of words in word_dict

    Args:
        path_emb (string): Folder path of downloaded glove file
        word_dict (dict): word dictionary
        word_embedding_dim: dimention of word embedding vectors

    Returns:
        numpy.ndarray, list: pretrained word embedding metrics, words can be found in glove files
    """
    print(len(word_dict))
    embedding_matrix = np.random.random((len(word_dict) + 1, word_embedding_dim))
    exist_word = []

    with open(glove_path, "r") as f:
        for line in tqdm(f):  # noqa: E741 ambiguous variable name 'l'
            # return l
            l = line.split()  # noqa: E741 ambiguous variable name 'l'
            word = l[0]
            # assert len(l) == 301, 'error'
            if len(l) != 301:
                continue
            if len(word) != 0:
                if word in word_dict:
                    try:
                        wordvec = [float(x) for x in l[1:]]
                    except Exception as e:
                        # import pdb;pdb.set_trace()
                        pass
                        print(line)
                        print(len(l))
                        print(word)
                        return 
                    index = word_dict[word]
                    embedding_matrix[index] = np.array(wordvec)
                    exist_word.append(word)

    return embedding_matrix, exist_word


# In[ ]:


word_dict = load_from_disk('./data/word_dict.pkl')
word_dict_all = load_from_disk('./data/word_dict_all.pkl')


# In[ ]:


len(word_dict)


# In[ ]:


glove_path = '/data2/yyr/dataset/glove.840B.300d.txt'
word_embedding_dim = 300
# a = load_glove_matrix(glove_path, word_dict, word_embedding_dim)
embedding_matrix, exist_word = load_glove_matrix(glove_path, word_dict, word_embedding_dim)
embedding_all_matrix, exist_all_word = load_glove_matrix(glove_path, word_dict_all, word_embedding_dim)


# In[ ]:


np.save(os.path.join('./data', 'embedding.npy'), embedding_matrix)
np.save(os.path.join('./data', 'embedding_all.npy'), embedding_all_matrix)


# In[ ]:


# len(exist_word)
print(len(exist_word))
print(len(word_dict))


# In[ ]:


exist_all_word[200:240]
words = [w for w in word_dict_all]
words = set(words)
exist = set(exist_all_word)
notin = list(words - exist)
# notin = list(exist)
print(len(notin))


# In[ ]:


notindi = {}
for i in notin:
    notindi[i] = word_cnt_all[i]
val = [v for k,v in notindi.items()]


# In[ ]:


val = np.array(val)
np.sum(val)


# ## uid2index

# In[5]:


person_path = '/home/zy/data2/yyr/codes/kg4proj-rec/OpenKS-Expert-Rec/data/datav1/entities_person.pkl'
person = load_from_disk(person_path)
uid2index = {}

for l in tqdm(person):
    uid = l
    if uid not in uid2index:
        uid2index[uid] = len(uid2index) + 1
save_to_disk(uid2index, './data/uid2index.pkl')


# ## 生成train和val test

# In[ ]:




