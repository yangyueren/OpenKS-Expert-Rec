import pdb
import pickle
import time
from tqdm import tqdm
import argparse
import hashlib
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import math

def load_pkl(path):
    print("Loading %s ..." % path)
    START = time.perf_counter()
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Time consumed [%s]: %.2f s" % (path, time.perf_counter() - START))
    return data


def gen_emb(ent, model):

    titles = []
    for t in ent:
        if 'title' not in t:
            t['title'] = ''
        titles.append(t['title'])
    abstracts = []
    for t in ent:
        if 'abstract' not in t:
            t['abstract'] = ''
        abstracts.append(t['abstract'])

    title = model.encode(titles)
    abstract = model.encode(abstracts)

    ids = [t['_id'] for t in ent]
    ans = []
    for id, ti, ab in zip(ids, title, abstract):
        ans.append((id, ti, ab))
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    assert args.model in ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1"]
    print("Loading model...")
    START = time.perf_counter()
    model = SentenceTransformer(args.model)
    print("Time consumed [%s]: %.2f s" % (args.model, time.perf_counter() - START))

    entities_paper = load_pkl(args.data_file)
    emb_res = []
    abs_num = 0
    batch_size = 1024
    for index in tqdm(range(math.ceil(len(entities_paper)/batch_size))):
        ents = entities_paper[index*batch_size: (index+1)*batch_size]
        embs = gen_emb(ents, model)
        emb_res += embs
        # pdb.set_trace()
    print("Abs / total : %d / %d" % (abs_num, len(entities_paper)))

    outfile = os.path.join(args.output_dir, "emb_paper_"+args.model.replace('-', '_')+'.pkl')

    print("Saving embeddings...")
    START = time.perf_counter()
    with open(outfile, "wb") as f:
        pickle.dump(emb_res, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Time consumed [%s]: %.2f s" % (outfile, time.perf_counter() - START))

    # print("Checking...")
    # START = time.perf_counter()
    # with open(outfile, "rb") as f:
    #     content = f.read()
    #     md5 = str(hashlib.md5(content).hexdigest())
    #     print(md5)
    # print("Time consumed [%s]: %.2f s" % (outfile, time.perf_counter() - START))
    # all-mpnet-base-v2:          1bdb571b4fa88408c204df25b6133e47
    # multi-qa-mpnet-base-dot-v1: 89e71bdfcdd541ec63cb26bad4db0eff


if __name__ == "__main__":
    main()
# python Generate_Embeddings/get_emb_paper.py --data_file data/datav1/entities_paper.pkl --output_dir data/datav1