import logging
import os
from pprint import pprint

import tensorflow as tf
from tensorflow import keras
from cqrconfig import parser
from data_helper import create_datasets
from cqrmetrics import Recorder
from cqrmodel_transformer import MultiModal_JT as MultiModal
from util import test_spearmanr

def contrastive_loss(projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        temperature = 0.07
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)
        similarities = (
            tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)
        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 =keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2


def train(args):
    # 1. create dataset and set num_labels to args
    train_dataset, val_dataset = create_datasets(args)
    # 2. build model
    model = MultiModal(args)
    # 3. save checkpoints
    checkpoint = tf.train.Checkpoint(model=model, step=tf.Variable(0))
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.savedmodel_path, args.max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        logging.info("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        logging.info("Initializing from scratch.")
    # 4. create loss_object and recorders
    loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    train_recorder, val_recorder = Recorder(), Recorder()

    # input_template = {
    #     "input_ids": tf.keras.layers.Input(
    #         shape=[args.bert_seq_length], batch_size=args.batch_size, dtype=tf.int64),
    #     "mask": tf.keras.layers.Input(
    #         shape=[args.bert_seq_length], batch_size=args.batch_size, dtype=tf.int64),
    #     'frames': tf.keras.layers.Input(
    #         shape=[args.max_frames, args.frame_embedding_size], batch_size=args.batch_size, dtype=tf.float32),
    #     'num_frames': tf.keras.layers.Input(
    #         shape=[], batch_size=args.batch_size, dtype=tf.int64),
    #     'vid': tf.keras.layers.Input(
    #         shape=[], batch_size=args.batch_size, dtype=tf.string),
    #     'labels': tf.keras.layers.Input(
    #         shape=[args.num_labels], batch_size=args.batch_size, dtype=tf.int64)
    # }
    # outputs_template = model(input_template)
    # eval_custom_model = MultiModal(args)
    # eval_outputs_template = eval_custom_model(input_template)
    # logging.info("Eval model outputs: {}.".format(eval_outputs_template))
    # 5. define train and valid step function
    @tf.function
    def train_step(inputs):
        labels = inputs['labels']
        with tf.GradientTape() as tape:
            predictions, _, vision_embedding, bert_embedding = model(inputs, training=True)
            loss_0 = loss_object(labels, predictions) * labels.shape[-1]  # convert mean back to sum
            loss_1 = contrastive_loss(vision_embedding, bert_embedding) * 10.0
            loss = loss_0 + loss_1
        gradients = tape.gradient(loss, model.get_variables())
        model.optimize(gradients)
        train_recorder.record(loss, loss_0, loss_1, labels, predictions)

    # 6. training
    for epoch in range(args.start_epoch, args.epochs):
        for train_batch in train_dataset:
            checkpoint.step.assign_add(1)
            step = checkpoint.step.numpy()
            if step > args.total_steps:
                break
            # if step == 1:
            #     evaluate(val_dataset, model, val_recorder, epoch, step, args)
            train_step(train_batch)
            if step % args.print_freq == 0:
                train_recorder.log(epoch, step)
                train_recorder.reset()

            # 7. validation
            if step % args.eval_freq == 0:
                spearmanr = evaluate(val_dataset, model, val_recorder, epoch, step, args)
                # 9. save checkpoints
                if spearmanr > 0.45:
                    checkpoint_manager.save(checkpoint_number=step)

@tf.function
def val_step(inputs, model, val_recorder):
    vids = inputs['vid']
    # tf.print(vids.shape)
    labels = inputs['labels']
    predictions, embeddings, vision_embedding, bert_embedding = model(inputs, training=False)
    loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss_0 = loss_object(labels, predictions) * labels.shape[-1]  # convert mean back to sum
    loss_1 = contrastive_loss(vision_embedding, bert_embedding) *10.0
    loss = loss_0 + loss_1
    val_recorder.record(loss,loss_0, loss_1, labels, predictions)
    return vids, embeddings

def evaluate(val_dataset, model, val_recorder, epoch, step, args):
    vid_embedding = {}
    for val_batch in val_dataset:
        vids, embeddings = val_step(val_batch, model, val_recorder)
        for vid, embedding in zip(vids.numpy(), embeddings.numpy()):
            vid = vid.decode('utf-8')
            vid_embedding[vid] = embedding
    # 8. test spearman correlation
    spearmanr = test_spearmanr(vid_embedding, args.annotation_file)
    val_recorder.log(epoch, step, prefix='Validation result is: ', suffix=f', spearmanr {spearmanr:.4f}')
    val_recorder.reset()
    return spearmanr

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
