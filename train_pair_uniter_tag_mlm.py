import logging
import os
from pprint import pprint

import tensorflow as tf

from config_pair import parser
from data_helper_pair_mask import create_datasets
from metrics_pair import Recorder_3
from model_pair_uniter import MultiModal_Uniter_mlm as MultiModal
import numpy as np
from scipy.stats import spearmanr
from cqrtrain_mlm_mm import contrastive_loss, compute_loss, shape_list


def MSE(sim, label):
    return tf.reduce_sum(tf.square(sim - label))
def KL(sim, label):
    return tf.keras.losses.KLDivergence()(label, sim)

def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    print(train_dataset)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step_1=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    restored_ckpt = tf.train.latest_checkpoint(args.pretrain_model_dir)
    checkpoint.restore(restored_ckpt).expect_partial()
    if restored_ckpt:
        logging.info("Restored from {}".format(restored_ckpt))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    loss_object = MSE
    loss_kl = tf.keras.losses.KLDivergence()
    loss_object_tag = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder_3(), Recorder_3()

    # 5. define train and valid step_1 function
    @tf.function
    def train_step_1(inputs):
        label_sims = inputs['sim']
        labels_1 = inputs['labels_1']
        labels_2 = inputs['labels_2']
        mask_labels_1 = inputs['mask_labels_1']
        mask_labels_2 = inputs['mask_labels_2']
        mask_labels = tf.concat([mask_labels_1,mask_labels_2],0)
        with tf.GradientTape() as tape:
            final_embedding_1, final_embedding_2, predictions_1, predictions_2,prediction_scores_mlm_1,prediction_scores_mlm_2 = model(inputs, training=True)
            final_embedding_1 = tf.math.l2_normalize(final_embedding_1, axis=1)
            final_embedding_2 = tf.math.l2_normalize(final_embedding_2, axis=1)
            sim = tf.reduce_sum(final_embedding_1 * final_embedding_2, axis=1)
            loss_0 = loss_object(sim, label_sims)
            #loss_1 = contrastive_loss(vision_embedding, bert_embedding) * 5.0
            predictions = tf.concat([predictions_1, predictions_2], 0)
            prediction_scores_mlm = tf.concat([prediction_scores_mlm_1,prediction_scores_mlm_2],0)
            labels = tf.concat([labels_1, labels_2], 0)
            loss_1 = loss_kl(label_sims, sim) 
            loss_tag = loss_object_tag(labels, predictions) * labels.shape[-1]  # convert mean back to sum
            loss_mlm = compute_loss(labels=mask_labels, logits=prediction_scores_mlm) * 5
            loss = loss_0 + loss_tag + args.kl_weight*loss_1 + loss_mlm#
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, loss_0, loss_1, loss_mlm)

    @tf.function
    def val_step_1(inputs):
        vids_1 = inputs['vid_1']
        vids_2 = inputs['vid_2']
        label_sims = inputs['sim']
        labels_1 = inputs['labels_1']
        labels_2 = inputs['labels_2']
        final_embedding_1, final_embedding_2, predictions_1, predictions_2,_,_ = model(inputs, training=False)
        final_embedding_1 = tf.math.l2_normalize(final_embedding_1, axis=1)
        final_embedding_2 = tf.math.l2_normalize(final_embedding_2, axis=1)
        sim = tf.reduce_sum(final_embedding_1 * final_embedding_2, axis=1)
        loss_0 = loss_object(sim, label_sims)
        #loss_1 = contrastive_loss(vision_embedding, bert_embedding) * 10.0
        predictions = tf.concat([predictions_1, predictions_2], 0)
        labels = tf.concat([labels_1, labels_2], 0)
        loss_1 = loss_kl(label_sims, sim)
        loss_tag = loss_object_tag(labels, predictions) * labels.shape[-1]  # convert mean back to sum
        loss = loss_0 + loss_tag + args.kl_weight*loss_1 #
        val_recorder.record(loss, loss_0, loss_1, loss_tag)
        return vids_1, sim, label_sims
    # import pdb;pdb.set_trace()
    # 6. training
    for epoch in range(args.start_epoch, args.epochs):
        for train_batch in train_dataset:

            checkpoint.step_1.assign_add(1)
            step_1 = checkpoint.step_1.numpy()
            # tf.print(step_1)

            if step_1 == 2:
                vids_ = []
                sims_ = []
                label_sims_ = []
                for val_batch in val_dataset:
                    vids, sims, label_sims = val_step_1(val_batch)
                    for vid, sim, label_sim in zip(vids.numpy(), sims.numpy(), label_sims.numpy()):
                        vids_.append(vid.decode('utf-8'))
                        sims_.append(sim)
                        label_sims_.append(label_sim)
                vids_, sims_, label_sims_ = np.array(vids_), np.array(sims_), np.array(label_sims_)
                # 8. test spearman correlation
                spearman = spearmanr(sims_, label_sims_)[0]
                val_recorder.log(epoch, step_1, prefix='Validation result is: ', suffix=f', spearmanr {spearman:.4f}')
                val_recorder.reset()

            # tf.print(args.total_steps)
            if step_1 > args.total_steps:
                break
            train_step_1(train_batch)
            if step_1 % args.print_freq == 0:
                train_recorder.log(epoch, step_1)
                train_recorder.reset()

            # 7. validation
            if step_1 % args.eval_freq == 0:
                vids_ = []
                sims_ = []
                label_sims_ = []
                for val_batch in val_dataset:
                    vids, sims, label_sims = val_step_1(val_batch)
                    for vid, sim, label_sim in zip(vids.numpy(), sims.numpy(), label_sims.numpy()):
                        vids_.append(vid.decode('utf-8'))
                        sims_.append(sim)
                        label_sims_.append(label_sim)
                vids_, sims_, label_sims_ = np.array(vids_), np.array(sims_), np.array(label_sims_)
                # 8. test spearman correlation
                spearman = spearmanr(sims_, label_sims_)[0]
                val_recorder.log(epoch, step_1, prefix='Validation result is: ', suffix=f', spearmanr {spearman:.4f}')
                val_recorder.reset()

                # 9. save checkpoints
                if spearman > 0.45:
                    checkpoint_manager.save(checkpoint_number=step_1)
    # last step
    vids_ = []
    sims_ = []
    label_sims_ = []
    for val_batch in val_dataset:
        vids, sims, label_sims = val_step_1(val_batch)
        for vid, sim, label_sim in zip(vids.numpy(), sims.numpy(), label_sims.numpy()):
            vids_.append(vid.decode('utf-8'))
            sims_.append(sim)
            label_sims_.append(label_sim)
    vids_, sims_, label_sims_ = np.array(vids_), np.array(sims_), np.array(label_sims_)
    # 8. test spearman correlation
    spearman = spearmanr(sims_, label_sims_)[0]
    val_recorder.log(epoch, step_1, prefix='Validation result is: ', suffix=f', spearmanr {spearman:.4f}')
    val_recorder.reset()
    checkpoint_manager.save(checkpoint_number=step_1)


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
