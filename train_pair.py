import logging
import os
from pprint import pprint

import tensorflow as tf

from config_pair import parser
from data_helper_pair import create_datasets
from metrics_pair import Recorder
from model_pair import MultiModal
import numpy as np
from scipy.stats import spearmanr


def MSE(sim, label):
    return tf.reduce_sum(tf.square(sim - label))


def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    restored_ckpt = tf.train.latest_checkpoint(args.pretrain_model_dir)
    checkpoint.restore(restored_ckpt).expect_partial()
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    loss_object = MSE
    train_recorder, val_recorder = Recorder(), Recorder()

    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        label_sims = inputs['sim']
        with tf.GradientTape() as tape:
            final_embedding_1, final_embedding_2 = model(inputs, training=True)
            final_embedding_1 = tf.math.l2_normalize(final_embedding_1, axis=1)
            final_embedding_2 = tf.math.l2_normalize(final_embedding_2, axis=1)
            sim = tf.reduce_sum(final_embedding_1 * final_embedding_2, axis=1)
            loss = loss_object(sim, label_sims)
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss)

    @tf.function
    def val_step(inputs):
        vids = inputs['vid']
        label_sims = inputs['sim']
        final_embedding_1, final_embedding_2 = model(inputs, training=True)
        final_embedding_1 = tf.math.l2_normalize(final_embedding_1, axis=1)
        final_embedding_2 = tf.math.l2_normalize(final_embedding_2, axis=1)
        sim = tf.reduce_sum(final_embedding_1 * final_embedding_2, axis=1)
        loss = loss_object(sim, label_sims)
        val_recorder.record(loss)
        return vids, sim, label_sims

    # 6. training
    for epoch in range(args.start_epoch, args.epochs):
        for train_batch in train_dataset:
            checkpoint.step.assign_add(1)
            step = checkpoint.step.numpy()
            if step > args.total_steps:
                break
            train_step(train_batch)
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()

            # 7. validation
            if step % args.eval_freq == 0:
                vids_ = []
                sims_ = []
                label_sims_ = []
                for val_batch in val_dataset:
                    vids, sims, label_sims = val_step(val_batch)
                    for vid, sim, label_sim in zip(vids.numpy(), sims.numpy(), label_sims.numpy()):
                        vids_.append(vid.decode('utf-8'))
                        sims_.append(sim)
                        label_sims_.append(label_sim)
                vids_, sims_, label_sims_ = np.array(vids_), np.array(sims_), np.array(label_sims_)
                # 8. test spearman correlation
                spearman = spearmanr(sims_, label_sims_)[0]
                val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearman:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearman > 0.45:
                    checkpoint_manager.save(checkpoint_number=step)


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    if not os.path.exists(args.savedmodel_path):
        os.makedirs(args.savedmodel_path)

    pprint(vars(args))
    train(args)


if __name__ == '__main__':
    main()
