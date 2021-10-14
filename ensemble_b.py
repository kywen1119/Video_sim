import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import tensorflow as tf

from config import parser
from data_helper import FeatureParser
from model import MultiModal


def uniter_mlm_tag_mean():
    output_json = 'result.json'
    output_zip = 'result_10fold_uniter_mlm_tag_mean_kl.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
        #                 np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]))/10).tolist()
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]))/8).tolist()
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + \
                        np.array(vid_emb_11[key]))/11).tolist() # 
                        
        # print(len(out_emb[key]))
        # exit()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)

def mix_addtf():
    output_json = 'result.json'
    output_zip = 'result_10fold_mix_addtf.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_mix_addtf.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_mix_addtf.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_mix_addtf.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_mix_addtf.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_mix_addtf.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_mix_addtf.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_mix_addtf.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_mix_addtf.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_mix_addtf.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_mix_addtf.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_mix_addtf.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
        #                 np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]))/10).tolist()
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) )/7).tolist()
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + \
                        np.array(vid_emb_11[key]))/11).tolist() # 
                        
        # print(len(out_emb[key]))
        # exit()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def uniter_mlm_tag_mean_filter():
    output_json = 'result.json'
    output_zip = 'result_filter_10fold_uniter_mlm_tag_mean_kl.zip'
    out_emb = {}
    with open('10fold_b_json/filter15_1_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/filter15_2_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/filter15_3_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/filter15_4_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/filter15_5_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/filter15_6_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/filter15_7_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/filter15_8_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/filter15_9_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/filter15_10_uniter_mlm_tag_mean_kl.json', 'r') as f:
        vid_emb_10 = json.load(f)
    # with open('10fold_b_json/10fold_11_uniter_mlm_tag_mean_kl.json', 'r') as f:
    #     vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
        #                 np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]))/10).tolist()
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]))/8).tolist()
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]))/10).tolist() # 
                        
        # print(len(out_emb[key]))
        # exit()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def mix_60():
    output_json = 'result.json'
    output_zip = 'result_10fold_mix_60.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_mix_60.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_mix_60.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_mix_60.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_mix_60.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_mix_60.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_mix_60.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_mix_60.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_mix_60.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_mix_60.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_mix_60.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_mix_60.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
        #                 np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]))/10).tolist()
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) )/7).tolist()
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + \
                        np.array(vid_emb_11[key]))/11).tolist() # 
                        
        # print(len(out_emb[key]))
        # exit()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def uniter_mlm_tag_mean_roformer():
    output_json = 'result.json'
    output_zip = 'result_10fold_uniter_mlm_tag_mean_kl_roformer.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_uniter_mlm_tag_mean_kl_roformer.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
        #                 np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]))/10).tolist()
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
        #                 np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
        #                 np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]))/8).tolist()
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + np.array(vid_emb_11[key]))/11).tolist() # 
                        
        # print(len(out_emb[key]))
        # exit()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)

if __name__ == '__main__':
    uniter_mlm_tag_mean_roformer()
