import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf

from config import parser
from data_helper import FeatureParser
from model import MultiModal


def main():
    output_json = 'ensemble/result.json'
    output_zip = 'result_ens.zip'
    out_emb = {}
    with open('result_pair_1.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('result_pair_2.json', 'r') as f:
        vid_emb_2 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = (np.array(vid_emb_1[key]) + np.array(vid_emb_2[key])).tolist()
        # print(len(out_emb[key]))
        # exit()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


if __name__ == '__main__':
    main()
