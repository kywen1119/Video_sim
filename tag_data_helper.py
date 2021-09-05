import glob
from util import *
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from transformers import BertTokenizer


class FeatureParser:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.max_bert_length = args.bert_seq_length
        self.max_frames = args.max_frames

        self.selected_tags = set()
        with tf.io.gfile.GFile(args.multi_label_file, 'r') as fh:
            for line in fh:
                tag_id = int(line.strip())
                self.selected_tags.add(tag_id)
        self.num_labels = len(self.selected_tags)
        args.num_labels = self.num_labels
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

    def _encode(self, title):
        title = title.numpy().decode(encoding='utf-8')
        encoded_inputs = self.tokenizer(title, max_length=self.max_bert_length, padding='max_length', truncation=True)
        input_ids = encoded_inputs['input_ids']
        mask = encoded_inputs['attention_mask']
        return input_ids, mask

    def _parse_title(self, title):
        input_ids, mask = tf.py_function(self._encode, [title], [tf.int32, tf.int32])
        input_ids.set_shape([self.max_bert_length])
        mask.set_shape([self.max_bert_length])
        return input_ids, mask

    def _sample(self, frames):
        # 这里采样逻辑是少于32帧 全部保留 并最后一帧补齐到32
        # 32 - 63帧 保留前32帧, 63帧以上 按间隔为2进行保留  以此类推
        frames = frames.numpy()
        frames_len = len(frames)
        num_frames = min(frames_len, self.max_frames)
        num_frames = np.array([num_frames], dtype=np.int32)

        average_duration = frames_len // self.max_frames
        if average_duration == 0:
            return [frames[min(i, frames_len - 1)] for i in range(self.max_frames)], num_frames
        else:
            offsets = np.multiply(list(range(self.max_frames)), average_duration) + average_duration // 2
            return [frames[i] for i in offsets], num_frames

    def _parse_frames(self, frames):
        frames = tf.sparse.to_dense(frames)
        frames, num_frames = tf.py_function(self._sample, [frames], [tf.string, tf.int32])
        frames_embedding = tf.map_fn(lambda x: tf.io.decode_raw(x, out_type=tf.float16), frames, dtype=tf.float16)
        frames_embedding = tf.cast(frames_embedding, tf.float32)
        frames_embedding.set_shape([self.max_frames, self.args.frame_embedding_size])
        num_frames.set_shape([1])
        return frames_embedding, num_frames

    def _parse_label(self, labels, cate_id):
        tags = labels.numpy()
        cate_id = cate_id.numpy()
        # tag filtering
        tags = [tag for tag in tags if tag in self.selected_tags]
        if cate_id in self.selected_tags:
            tags.append(cate_id)
        multi_hot = self.mlb.transform([tags])[0].astype(dtype=np.int8)
        return tf.convert_to_tensor(multi_hot)

    def _parse_labels(self, labels, category_id):
        labels = tf.sparse.to_dense(labels)
        labels = tf.py_function(self._parse_label, [labels, category_id], [tf.int8])[0]
        labels.set_shape([self.num_labels])
        return labels

    def parse(self, features):
        input_ids, mask = self._parse_title(features['title'])  #
        frames, num_frames = self._parse_frames(features['frame_feature'])  # [32, 1536] embedding
        labels = self._parse_labels(features['tag_id'], features['category_id'])  # multi-hot encoding tag&category
        return {'input_ids': input_ids, 'mask': mask, 'frames': frames, 'num_frames': num_frames,
                'vid': features['id'], 'labels': labels}

    def create_dataset(self, files, training, batch_size):
        if training:
            np.random.shuffle(files)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
        feature_map = {'id': tf.io.FixedLenFeature([], tf.string),
                       'title': tf.io.FixedLenFeature([], tf.string),
                       'frame_feature': tf.io.VarLenFeature(tf.string),
                       'tag_id': tf.io.VarLenFeature(tf.int64),
                       'category_id': tf.io.FixedLenFeature([], tf.int64)
                       }
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map), num_parallel_calls=AUTOTUNE)
        if training:
            dataset = dataset.shuffle(buffer_size=batch_size * 8)
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=training)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset


def create_datasets(args):
    train_files = tf.io.gfile.glob(args.train_record_pattern)
    val_files = tf.io.gfile.glob(args.val_record_pattern)

    parser = FeatureParser(args)
    train_dataset = parser.create_dataset(train_files, training=True, batch_size=args.batch_size)
    val_dataset = parser.create_dataset(val_files, training=False, batch_size=args.val_batch_size)

    return train_dataset, val_dataset