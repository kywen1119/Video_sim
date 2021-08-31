import logging
import tensorflow as tf


class Recorder:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()
        self.loss0 = tf.keras.metrics.Mean()
        self.loss1 = tf.keras.metrics.Mean()
        self.loss_cate_1 = tf.keras.metrics.Mean()
        self.loss_cate_2 = tf.keras.metrics.Mean()
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.acc_1 = tf.keras.metrics.SparseCategoricalAccuracy()
        self.acc_2 = tf.keras.metrics.SparseCategoricalAccuracy()

        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}, loss_0: {:.4f}, loss_1: {:.4f}, loss_cate_1: {:.4f}, loss_cate_2: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}, acc_1: {:.4f}, acc_2: {:.4f}'

    def record(self, losses, loss_0, loss_1, loss_cate_1, loss_cate_2, labels, predictions, labels_cate_1, pre_1, labels_cate_2, pre_2):
        self.loss.update_state(losses)
        self.loss0.update_state(loss_0)
        self.loss1.update_state(loss_1)
        self.loss_cate_1.update_state(loss_cate_1)
        self.loss_cate_2.update_state(loss_cate_2)
        self.precision.update_state(labels, predictions)
        self.recall.update_state(labels, predictions)
        self.acc_1.update_state(labels_cate_1, pre_1)
        self.acc_2.update_state(labels_cate_2, pre_2)

    def reset(self):
        self.loss.reset_states()
        self.loss0.reset_states()
        self.loss1.reset_states()
        self.loss_cate_1.reset_states()
        self.loss_cate_2.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.acc_1.reset_states()
        self.acc_2.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        loss0 = self.loss0.result().numpy()
        loss1 = self.loss1.result().numpy()
        loss_cate_1 = self.loss_cate_1.result().numpy()
        loss_cate_2 = self.loss_cate_2.result().numpy()
        precision = self.precision.result().numpy()
        recall = self.recall.result().numpy()
        f1 = 2 * precision * recall / (precision + recall + 1e-6)  # avoid division by 0
        acc_1 = self.acc_1.result().numpy()
        acc_2 = self.acc_2.result().numpy()
        return [loss, loss0, loss1, loss_cate_1, loss_cate_2, precision, recall, f1, acc_1, acc_2]

    def score(self):
        return self._results()[-3].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss, loss0, loss1, loss_cate_1, loss_cate_2, precision, recall, f1, acc_1, acc_2 = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss, loss0, loss1, loss_cate_1, loss_cate_2, precision, recall, f1, acc_1, acc_2) + suffix)
