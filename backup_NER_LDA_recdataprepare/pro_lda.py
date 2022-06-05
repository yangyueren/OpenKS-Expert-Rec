r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

###############################################################################
# The purpose of this tutorial is to demonstrate how to train and tune an LDA model.
#
# In this tutorial we will:
#
# * Load input data.
# * Pre-process that data.
# * Transform documents into bag-of-words vectors.
# * Train an LDA model.
#
# This tutorial will **not**:
#
# * Explain how Latent Dirichlet Allocation works
# * Explain how the LDA model performs inference
# * Teach you all the parameters and options for Gensim's LDA implementation
#
# If you are not familiar with the LDA model or how to use it in Gensim, I (Olavur Mortensen)
# suggest you read up on that before continuing with this tutorial. Basic
# understanding of the LDA model should suffice. Examples:
#
# * `Introduction to Latent Dirichlet Allocation <http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation>`_
# * Gensim tutorial: :ref:`sphx_glr_auto_examples_core_run_topics_and_transformations.py`
# * Gensim's LDA model API docs: :py:class:`gensim.models.LdaModel`
#
# I would also encourage you to consider each step when applying the model to
# your data, instead of just blindly applying my solution. The different steps
# will depend on your data and possibly your goal with the model.
#
# Data
# ----
#
# I have used a corpus of NIPS papers in this tutorial, but if you're following
# this tutorial just to learn about LDA I encourage you to consider picking a
# corpus on a subject that you are familiar with. Qualitatively evaluating the
# output of an LDA model is challenging and can require you to understand the
# subject matter of your corpus (depending on your goal with the model).
#
# NIPS (Neural Information Processing Systems) is a machine learning conference
# so the subject matter should be well suited for most of the target audience
# of this tutorial.  You can download the original data from Sam Roweis'
# `website <http://www.cs.nyu.edu/~roweis/data.html>`_.  The code below will
# also do that for you.
#
# .. Important::
#     The corpus contains 1740 documents, and not particularly long ones.
#     So keep in mind that this tutorial is not geared towards efficiency, and be
#     careful before applying the code to a large dataset.
#

import io
import os.path
import re
import tarfile
import copy
from tqdm import tqdm
import smart_open

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
content, awardids = get_docs()



first_run = False
if first_run:
    print('first_run is True')
    print('begin read docs')
    def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
        with open('nips12raw_str602.tgz', "rb") as file:
            with tarfile.open(fileobj=file) as tar:
                for member in tar.getmembers():
                    if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                        member_bytes = tar.extractfile(member).read()
                        yield member_bytes.decode('utf-8', errors='replace')

    

    # docs = list(extract_documents())
    docs, awardids = get_docs()

    ###############################################################################
    # So we have a list of 1740 documents, where each document is a Unicode string.
    # If you're thinking about using your own corpus, then you need to make sure
    # that it's in the same format (list of Unicode strings) before proceeding
    # with the rest of this tutorial.
    #
    print(len(docs))
    print(docs[0][:500])
    content = copy.deepcopy(docs)

    ###############################################################################
    # Pre-process and vectorize the documents
    # ---------------------------------------
    #
    # As part of preprocessing, we will:
    #
    # * Tokenize (split the documents into tokens).
    # * Lemmatize the tokens.
    # * Compute bigrams.
    # * Compute a bag-of-words representation of the data.
    #
    # First we tokenize the text using a regular expression tokenizer from NLTK. We
    # remove numeric tokens and tokens that are only a single character, as they
    # don't tend to be useful, and the dataset contains a lot of them.
    #
    # .. Important::
    #
    #    This tutorial uses the nltk library for preprocessing, although you can
    #    replace it with something else if you want.
    #

    # Tokenize the documents.
    from nltk.tokenize import RegexpTokenizer

    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]


    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    def stemming(tokens):
        stemmer= PorterStemmer()
        
        # tokens = word_tokenize(str(data))
        new_text = []
        for w in tokens:
            new_text.append(stemmer.stem(w))
        # import pdb;pdb.set_trace()
        return new_text

    docs = [stemming(doc) for doc in docs]

    stopwords = set()
    with open('stopwords.txt', 'r') as f:
        for w in f.readlines():
            w = w.strip()
            stopwords.add(w)
    docs = [[token for token in doc if token not in stopwords] for doc in docs]


    ###############################################################################
    # We use the WordNet lemmatizer from NLTK. A lemmatizer is preferred over a
    # stemmer in this case because it produces more readable words. Output that is
    # easy to read is very desirable in topic modelling.
    #

    # Lemmatize the documents.
    # from nltk.stem.wordnet import WordNetLemmatizer

    # lemmatizer = WordNetLemmatizer()
    # docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    save_to_disk(docs, './docs.pkl')

if first_run is False:
    print('first_run is False')
    
docs = load_from_disk('./docs.pkl')
stopwords = set()
print('loading stopwords.txt...')
with open('stopwords.txt', 'r') as f:
    for w in f.readlines():
        w = w.strip()
        stopwords.add(w)
docs = [[token for token in doc if token not in stopwords] for doc in docs]

###############################################################################
# We find bigrams in the documents. Bigrams are sets of two adjacent words.
# Using bigrams we can get phrases like "machine_learning" in our output
# (spaces are replaced with underscores); without bigrams we would only get
# "machine" and "learning".
#
# Note that in the code below, we find bigrams and then add them to the
# original data, because we would like to keep the words "machine" and
# "learning" as well as the bigram "machine_learning".
#
# .. Important::
#     Computing n-grams of large dataset can be very computationally
#     and memory intensive.
#


# Compute bigrams.
from gensim.models import Phrases

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
# bigram = Phrases(docs, min_count=20)
# for idx in range(len(docs)):
#     for token in bigram[docs[idx]]:
#         if '_' in token:
#             # Token is a bigram, add to document.
#             docs[idx].append(token)

###############################################################################
# We remove rare words and common words based on their *document frequency*.
# Below we remove words that appear in less than 20 documents or in more than
# 50% of the documents. Consider trying to remove words only based on their
# frequency, or maybe combining that with this approach.
#

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)
# dictionary.filter_extremes(no_below=2, no_above=0.5)

###############################################################################
# Finally, we transform the documents to a vectorized form. We simply compute
# the frequency of each word, including the bigrams.
#

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

assert len(corpus) == len(content), 'error'
###############################################################################
# Let's see how many tokens and documents we have to train on.
#

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


predict = True
from gensim.models import LdaModel
import numpy as np
if predict:
    # inference
    path = './saved/lda_passes40.model'
    print('predict is True. Will load model from ', path)
    model = LdaModel.load(path)
    project2topic = []
    
    for e, values in enumerate(model.inference(corpus)[0]):
        awardid = awardids[e]
        val = np.array(values)
        idx = val.argmax()
        project2topic.append((awardid, idx))
    save_to_disk(project2topic, './project2topic.pkl')
    with open('./topics.txt', 'w') as f:
        for l in model.print_topics(num_words=20):
            f.write(str(l[0]))
            f.write('\n')
            f.write(l[1])
            f.write('\n')

    exit(0)

# Train LDA model.
from gensim.models import LdaModel

for i in range(1,11):

    # Set training parameters.
    num_topics = 20
    chunksize = 2000
    passes = 10 * i
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

    print(f'passes is {passes}')
    # import pdb; pdb.set_trace()
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        # chunksize=chunksize,
        # alpha='auto',
        # eta='auto',
        # iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        # eval_every=eval_every
    )

    top_topics = model.top_topics(corpus)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)

    print('*'*20)
    for topic in model.print_topics(num_words=10):
        print(topic)
    
    print(f'./saved/lda_passes{passes}.model')
    model.save(f'./saved/lda_passes{passes}.model')


    # inference
    #for e, values in enumerate(model.inference(corpus)[0]):
        #print(content[e][:50])
        #for ee, value in enumerate(values):
        #    print('\t主题%d推断值%.2f' % (ee, value))
        #if e > 4:
         #   break
