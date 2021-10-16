import json
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np
import os


def mix():
    output_json = '10_b/result_10fold_mix.json'
    output_zip = 'result_10fold_mix.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_mix.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_mix.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_mix.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_mix.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_mix.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_mix.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_mix.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_mix.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_mix.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_mix.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_mix.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + np.array(vid_emb_11[key]))/11).tolist()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def mix_asl(): 
    output_json = '10_b/result_10fold_mix_asl.json'
    output_zip = 'result_10fold_mix_asl.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_mix_asl.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_mix_asl.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_mix_asl.json', 'r') as f:
        vid_emb_3 = json.load(f)
    # with open('10fold_b_json/10fold_4_mix_asl.json', 'r') as f:
    #     vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_mix_asl.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_mix_asl.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_mix_asl.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_mix_asl.json', 'r') as f:
        vid_emb_8 = json.load(f)
    # with open('10fold_b_json/10fold_9_mix_asl.json', 'r') as f:
    #     vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_mix_asl.json', 'r') as f:
        vid_emb_10 = json.load(f)
    # with open('10fold_b_json/10fold_11_mix_asl.json', 'r') as f:
    #     vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + np.array(vid_emb_3[key]) + \
                        np.array(vid_emb_5[key])+ np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_10[key]))/8).tolist()
        # out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + np.array(vid_emb_3[key]) + \
        #                 np.array(vid_emb_4[key]) + np.array(vid_emb_5[key])+ np.array(vid_emb_6[key]) + \
        #                 np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + np.array(vid_emb_9[key]) + \
        #                 np.array(vid_emb_10[key]) + np.array(vid_emb_11[key]))/11).tolist()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def mix_roformer():
    output_json = '10_b/result_10fold_mix_roformer.json'
    output_zip = 'result_10fold_mix_roformer.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_mix_roformer.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_mix_roformer.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_mix_roformer.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_mix_roformer.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_mix_roformer.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_mix_roformer.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_mix_roformer.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_mix_roformer.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_mix_roformer.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_mix_roformer.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_mix_roformer.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + np.array(vid_emb_3[key]) + \
                        np.array(vid_emb_4[key])+ np.array(vid_emb_5[key])+ np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key])+ np.array(vid_emb_8[key])+ np.array(vid_emb_9[key]) + \
                        np.array(vid_emb_10[key]) + np.array(vid_emb_11[key]))/11).tolist()
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def uniter():
    output_json = '10_b/result_10fold_uniter.json'
    output_zip = 'result_10fold_uniter.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_uniter.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_uniter.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_uniter.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_uniter.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_uniter.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_uniter.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_uniter.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_uniter.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_uniter.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_uniter.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_uniter.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + \
                        np.array(vid_emb_11[key]))/11).tolist() # 
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def uniter_asl():
    output_json = '10_b/result_10fold_uniter_asl.json'
    output_zip = 'result_10fold_uniter_asl.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_uniter_asl.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_uniter_asl.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_uniter_asl.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_uniter_asl.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_uniter_asl.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_uniter_asl.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_uniter_asl.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_uniter_asl.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_uniter_asl.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_uniter_asl.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_uniter_asl.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + \
                        np.array(vid_emb_11[key]))/11).tolist() # 
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


def uniter_roformer():
    output_json = '10_b/result_10fold_uniter_roformer.json'
    output_zip = 'result_10fold_uniter_roformer.zip'
    out_emb = {}
    with open('10fold_b_json/10fold_1_uniter_roformer.json', 'r') as f:
        vid_emb_1 = json.load(f)
    with open('10fold_b_json/10fold_2_uniter_roformer.json', 'r') as f:
        vid_emb_2 = json.load(f)
    with open('10fold_b_json/10fold_3_uniter_roformer.json', 'r') as f:
        vid_emb_3 = json.load(f)
    with open('10fold_b_json/10fold_4_uniter_roformer.json', 'r') as f:
        vid_emb_4 = json.load(f)
    with open('10fold_b_json/10fold_5_uniter_roformer.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10fold_b_json/10fold_6_uniter_roformer.json', 'r') as f:
        vid_emb_6 = json.load(f)
    with open('10fold_b_json/10fold_7_uniter_roformer.json', 'r') as f:
        vid_emb_7 = json.load(f)
    with open('10fold_b_json/10fold_8_uniter_roformer.json', 'r') as f:
        vid_emb_8 = json.load(f)
    with open('10fold_b_json/10fold_9_uniter_roformer.json', 'r') as f:
        vid_emb_9 = json.load(f)
    with open('10fold_b_json/10fold_10_uniter_roformer.json', 'r') as f:
        vid_emb_10 = json.load(f)
    with open('10fold_b_json/10fold_11_uniter_roformer.json', 'r') as f:
        vid_emb_11 = json.load(f)
    for key in vid_emb_1:
        out_emb[key] = ((np.array(vid_emb_1[key]) + np.array(vid_emb_2[key]) + \
                        np.array(vid_emb_3[key]) + np.array(vid_emb_4[key]) + \
                        np.array(vid_emb_5[key]) + np.array(vid_emb_6[key]) + \
                        np.array(vid_emb_7[key]) + np.array(vid_emb_8[key]) + \
                        np.array(vid_emb_9[key]) + np.array(vid_emb_10[key]) + np.array(vid_emb_11[key]))/11).tolist() 
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)

def ten():
    output_json = 'result.json'
    output_zip = 'result_10_b.zip'
    out_emb = {}

    with open('10_b/result_10fold_mix.json', 'r') as f: 
        vid_emb_1 = json.load(f)
    with open('10_b/result_10fold_mix_asl.json', 'r') as f: 
        vid_emb_2 = json.load(f)
    with open('10_b/result_10fold_mix_roformer.json', 'r') as f: 
        vid_emb_3 = json.load(f)
    with open('10_b/result_10fold_uniter.json', 'r') as f: 
        vid_emb_4 = json.load(f)
    with open('10_b/result_10fold_uniter_asl.json', 'r') as f:
        vid_emb_5 = json.load(f)
    with open('10_b/result_10fold_uniter_roformer.json', 'r') as f: 
        vid_emb_6 = json.load(f)

    for key in vid_emb_1:
        out_emb[key] = (0.17*np.array(vid_emb_1[key]) + \
                        0.2*np.array(vid_emb_2[key]) + 0.13*np.array(vid_emb_3[key]) +\
                        0.13*np.array(vid_emb_4[key])+0.2*np.array(vid_emb_5[key]) + \
                        0.17*np.array(vid_emb_6[key])).tolist() 
    with open(output_json, 'w') as f:
        json.dump(out_emb, f)
    with ZipFile(output_zip, 'w', compression=ZIP_DEFLATED) as zip_file:
        zip_file.write(output_json)


if __name__ == '__main__':
    if not os.path.exists('10_b'):
        os.mkdir('10_b')
    print('10fold mix')
    mix()
    print('10fold mix_asl')
    mix_asl()
    print('10fold mix_roformer')
    mix_roformer()
    print('10fold uniter')
    uniter()
    print('10fold uniter_asl')
    uniter_asl()
    print('10fold uniter_roformer')
    uniter_roformer()
    print('6 model ensemble')
    ten()