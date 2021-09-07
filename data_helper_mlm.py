import glob
import logging

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from transformers import BertTokenizer
from config import parser


class FeatureParser:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
        self.max_bert_length = args.bert_seq_length
        self.max_frames = args.max_frames

        self.selected_tags = set()
        with open(args.multi_label_file, encoding='utf-8') as fh:
            for line in fh:
                tag_id = int(line.strip())
                self.selected_tags.add(tag_id)
        self.num_labels = len(self.selected_tags)
        args.num_labels = self.num_labels
        logging.info('Num of selected supervised qeh tags is {}'.format(self.num_labels))
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

    def mask_tokens(self, inputs, mlm_probability, tokenizer, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = np.random.random_sample(labels.shape)
        special_tokens_mask = special_tokens_mask.astype(np.bool_)

        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = probability_matrix > (1 - mlm_probability)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (np.random.random_sample(labels.shape) < 0.8) & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (np.random.random_sample(labels.shape) < 0.5) & masked_indices & ~indices_replaced
        random_words = np.random.randint(low=0, high=len(tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64)
        inputs[indices_random] = random_words

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _encode(self, title):
        mlm_probability = 0.15
        title = title.numpy().decode(encoding='utf-8')
        encoded_inputs = self.tokenizer(title, max_length=self.max_bert_length, padding='max_length', truncation=True, return_special_tokens_mask=True)
        example = self.tokenizer.pad(encoded_inputs, return_tensors="np", pad_to_multiple_of=None)
        special_tokens_mask = example.pop("special_tokens_mask", None)
        example["input_ids"], example["labels"] = self.mask_tokens(
            example["input_ids"], mlm_probability, self.tokenizer, special_tokens_mask=special_tokens_mask)
        if self.tokenizer.pad_token_id is not None:
            example["labels"][example["labels"] == self.tokenizer.pad_token_id] = -100
        example = {key: tf.convert_to_tensor(arr) for key, arr in example.items()}

        input_ids = example['input_ids']
        mask = example['attention_mask']
        mask_labels = example['labels']
        return input_ids, mask, mask_labels

    def _parse_title(self, title):
        input_ids, mask, mask_labels = tf.py_function(self._encode, [title], [tf.int32, tf.int32])
        input_ids.set_shape([self.max_bert_length])
        mask.set_shape([self.max_bert_length])
        mask_labels.set_shape([self.max_bert_length])
        return input_ids, mask, mask_labels

    def _sample(self, frames):
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

    def _parse_label(self, labels):
        tags = labels.numpy()
        # tag filtering
        tags = [tag for tag in tags if tag in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0].astype(dtype=np.int8)
        return tf.convert_to_tensor(multi_hot)

    def _parse_labels(self, labels):
        labels = tf.sparse.to_dense(labels)
        labels = tf.py_function(self._parse_label, [labels], [tf.int8])[0]
        labels.set_shape([self.num_labels])
        return labels

    def _parse_category(self, category_id):
        category_id = str(category_id.numpy())
        id_1 = int(category_id[1:3])
        if id_1 > 37:
            id_1 = id_1 - 2
        id_2 = int(category_id[3:])
        return id_1, id_2

    def parse(self, features):
        input_ids, mask, mask_labels = self._parse_title(features['title'])
        frames, num_frames = self._parse_frames(features['frame_feature'])
        labels = self._parse_labels(features['tag_id'])
        # category_id_1, category_id_2 = tf.py_function(self._parse_category, [features['category_id']], [tf.int8, tf.int8])
        return {'input_ids': input_ids, 'mask': mask, 'frames': frames, 'num_frames': num_frames,
                'vid': features['id'], 'labels': labels, 'mask_labels': mask_labels}#, 'category_id_1': category_id_1, 'category_id_2': category_id_2}

    def create_dataset(self, files, training, batch_size):
        if training:
            np.random.shuffle(files)
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
        feature_map = {'id': tf.io.FixedLenFeature([], tf.string),
                       'title': tf.io.FixedLenFeature([], tf.string),
                       'frame_feature': tf.io.VarLenFeature(tf.string),
                       'tag_id': tf.io.VarLenFeature(tf.int64)}
                    #    'category_id': tf.io.FixedLenFeature([], tf.int64)}
        dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_map), num_parallel_calls=AUTOTUNE)
        if training:
            dataset = dataset.shuffle(buffer_size=batch_size * 8)
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=training)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset


def create_datasets(args):
    train_files = glob.glob(args.train_record_pattern)
    val_files = glob.glob(args.val_record_pattern)

    parser = FeatureParser(args)
    train_dataset = parser.create_dataset(train_files, training=True, batch_size=args.batch_size)
    val_dataset = parser.create_dataset(val_files, training=False, batch_size=args.val_batch_size)

    return train_dataset, val_dataset


if __name__ == '__main__':
    args = parser.parse_args()
    train_dataset, val_dataset = create_datasets(args)
    for i in train_dataset:
        print(i['category_id_1'], i['category_id_2'])
        break