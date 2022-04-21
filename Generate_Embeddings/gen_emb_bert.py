
import pdb
import pickle
import time
from tqdm import tqdm
import argparse
import hashlib
import os
import numpy as np
import json
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_pkl(path):
    print("Loading %s ..." % path)
    START = time.perf_counter()
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Time consumed [%s]: %.2f s" % (path, time.perf_counter() - START))
    return data


def gen_emb(ent, model, tokenizer):
    title = model(**tokenizer(ent["title"], return_tensors='pt'))[0][0][0] if len(ent["title"]) > 0 else None
    abstract = model(**tokenizer(ent["abstract"], return_tensors='pt'))[0][0][0] if len(ent["abstract"]) > 0 else None
    return (ent["abstract"], title, abstract)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-cased")
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    # assert args.model in ["bert-base-cased"]
    print("Loading model...")
    START = time.perf_counter()
    model = AutoModel.from_pretrained(args.model)
    print("Time consumed [%s]: %.2f s" % (args.model, time.perf_counter() - START))

    entities_project = load_pkl(args.data_file)
    emb_res = []
    abs_num = 0
    for ent in tqdm(entities_project):
        # ent = json.loads(ent)
        # pdb.set_trace()
        emb = gen_emb(ent, model)
        if emb[2] is not None:
            abs_num += 1
        emb_res.append(emb)
        # pdb.set_trace()
    print("Abs / total : %d / %d" % (abs_num, len(entities_project)))

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
# python Generate_Embeddings/gen_emb_bert.py --model TwoTowerRec/pretrained_model/bert-base-cased --data_file data/datav1/entities_paper.pkl --output_dir data/datav1