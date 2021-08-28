import tensorflow as tf


feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'id': tf.io.FixedLenFeature([], tf.string),
    'tag_id': tf.io.VarLenFeature(tf.int64),
    'category_id': tf.io.FixedLenFeature([], tf.int64),
    'title': tf.io.FixedLenFeature([], tf.string),
    'asr_text': tf.io.FixedLenFeature([], tf.string),
    'frame_feature': tf.io.VarLenFeature(tf.string)
}


def read_and_decode(example_string):
    '''
    从TFrecord格式文件中读取数据 train
    '''
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    frame_feature = tf.sparse.to_dense(feature_dict['frame_feature']).numpy()
    title = feature_dict['title'].numpy()
    asr_text = feature_dict['asr_text'].numpy()
    id = feature_dict['id'].numpy()
    tag_id = tf.sparse.to_dense(feature_dict['tag_id']).numpy()
    category_id = feature_dict['category_id'].numpy()


    return id, tag_id, category_id, frame_feature, title, asr_text

import glob
def get_all_data(path): # 'data/pairwise'
    filenames = glob.glob(path)
    print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)
    datas = {}
    for i, data in enumerate(dataset):
        id, tag_id, category_id, frame_feature, title, asr_text = read_and_decode(data)
        id = id.decode()
        datas[id] = {'tag_id': tag_id, 'category_id': category_id, 'frame_feature': frame_feature, 'title': title, 'asr_text': asr_text}
        # print(id)
        # print(datas['2345203561710400875']['asr_text'])
        # break
        # if i % 10000 == 0 and i > 0:
        #     break
    return datas  

datas = get_all_data('data/pairwise/pairwise.tfrecords')

label_path = 'data/pairwise/label.tsv'
f = open(label_path)
all_pair_data = []
for line in f:
    id_1, id_2, sim = line.strip().split('\t')
    sim = float(sim)
    all_pair_data.append([id_1, id_2, sim])

# shuffle pair data and get the top 6000 for validation
import random

random.seed(42)
# print(all_pair_data[:10])
random.shuffle(all_pair_data)
# print(all_pair_data[:10])
val_pair_data = all_pair_data[:6000]
train_pair_data = all_pair_data[6000:]

from tqdm import tqdm

def write_tfrecord(pair_datas, split):
    write_path = 'data/pairwise/0-5999val/'+split+'.tfrecord'
    writer = tf.io.TFRecordWriter(write_path) 
    for pair_data in tqdm(pair_datas): # [id_1, id_2, sim] [str, str, float]
        id_1, id_2, sim = pair_data
        tag_id_1 = datas[id_1]['tag_id']
        category_id_1 = datas[id_1]['category_id']
        frame_feature_1 = datas[id_1]['frame_feature'].tolist()
        title_1 = datas[id_1]['title']
        asr_text_1 = datas[id_1]['asr_text']

        tag_id_2 = datas[id_2]['tag_id']
        category_id_2 = datas[id_2]['category_id']
        frame_feature_2 = datas[id_2]['frame_feature'].tolist()
        title_2 = datas[id_2]['title']
        asr_text_2 = datas[id_2]['asr_text']
        feature = {                             # 建立 tf.train.Feature 字典
            'id_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(id_1.encode())])),  
            'tag_id_1': tf.train.Feature(int64_list=tf.train.Int64List(value=list(tag_id_1))),
            'frame_feature_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=frame_feature_1)),
            'category_id_1': tf.train.Feature(int64_list=tf.train.Int64List(value=[category_id_1])),   
            'title_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[title_1])),
            'asr_text_1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[asr_text_1])),
            'id_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes(id_2.encode())])),  
            'tag_id_2': tf.train.Feature(int64_list=tf.train.Int64List(value=list(tag_id_2))),
            'frame_feature_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=frame_feature_2)),
            'category_id_2': tf.train.Feature(int64_list=tf.train.Int64List(value=[category_id_2])),   
            'title_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[title_2])),
            'asr_text_2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[asr_text_2])),
            'sim': tf.train.Feature(float_list=tf.train.FloatList(value=[sim])) 
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString()) 
    writer.close()


write_tfrecord(val_pair_data, 'val')
write_tfrecord(train_pair_data, 'train')