import pickle
import time
from tqdm import tqdm
import argparse
import hashlib
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer


def load_pkl(path):
    print("Loading %s ..." % path)
    START = time.perf_counter()
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Time consumed [%s]: %.2f s" % (path, time.perf_counter() - START))
    return data


def gen_emb(ent, model):
    return (ent["AwardID"], model.encode([ent["AwardTitle"]])[0])


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

    entities_project = load_pkl(args.data_file)
    emb_res = []
    for ent in tqdm(entities_project):
        ent = json.loads(ent)
        emb = gen_emb(ent, model)
        emb_res.append(emb)

    outfile = os.path.join(args.output_dir, "project_emb_"+args.model.replace('-', '_')+'.pkl')

    print("Saving embeddings...")
    START = time.perf_counter()
    with open(outfile, "wb") as f:
        pickle.dump(emb_res, f)
    print("Time consumed [%s]: %.2f s" % (outfile, time.perf_counter() - START))

    print("Checking...")
    START = time.perf_counter()
    with open(outfile, "rb") as f:
        content = f.read()
        md5 = str(hashlib.md5(content).hexdigest())
        print(md5)
    print("Time consumed [%s]: %.2f s" % (outfile, time.perf_counter() - START))
    # all-mpnet-base-v2:          ec491da56366e8d26ecb85cb9f7ddb58
    # multi-qa-mpnet-base-dot-v1: 2b49f735a9cc4446ebd515c9f4a01b7b


if __name__ == "__main__":
    main()
