import tensorflow as tf

from config_pair import parser
from data_helper_pair import FeatureParser
from model_pair import MultiModal
import numpy as np
from scipy.stats import spearmanr


def val_step_1(inputs, model):
    vids_1 = inputs['vid_1']
    vids_2 = inputs['vid_2']
    label_sims = inputs['sim']
    labels_1 = inputs['labels_1']
    labels_2 = inputs['labels_2']
    final_embedding_1, final_embedding_2, predictions_1, predictions_2 = model(inputs, training=False)
    final_embedding_1 = tf.math.l2_normalize(final_embedding_1, axis=1)
    final_embedding_2 = tf.math.l2_normalize(final_embedding_2, axis=1)
    sim = tf.reduce_sum(final_embedding_1 * final_embedding_2, axis=1)
    return vids_1, sim, label_sims

def main(args):
    files = args.val_record_pattern
    dataset = feature_parser.create_dataset(files, training=False, batch_size=args.test_batch_size)
    model = MultiModal(args)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args.ckpt_file)
    print(f"Restored from {args.ckpt_file}")
    vids_ = []
    sims_ = []
    label_sims_ = []
    for val_batch in dataset:
        vids, sims, label_sims = val_step_1(val_batch, model)
        for vid, sim, label_sim in zip(vids.numpy(), sims.numpy(), label_sims.numpy()):
            vids_.append(vid.decode('utf-8'))
            sims_.append(sim)
            label_sims_.append(label_sim)
    vids_, sims_, label_sims_ = np.array(vids_), np.array(sims_), np.array(label_sims_)
    # 8. test spearman correlation
    spearman = spearmanr(sims_, label_sims_)[0]
    print('spearman: %4f' % spearman)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

