import logging
import tensorflow as tf


class Recorder:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()
        self.loss_0 = tf.keras.metrics.Mean()
        self.loss_1 = tf.keras.metrics.Mean()

        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}, loss_0: {:.4f}, loss_1: {:.4f}'

    def record(self, losses, losses_0, losses_1):
        self.loss.update_state(losses)
        self.loss_0.update_state(losses_0)
        self.loss_1.update_state(losses_1)

    def reset(self):
        self.loss.reset_states()
        self.loss_0.reset_states()
        self.loss_1.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        loss_0 = self.loss_0.result().numpy()
        loss_1 = self.loss_1.result().numpy()
        return [loss, loss_0, loss_1]

    def score(self):
        return self._results()[0].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss, loss_0, loss_1 = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss, loss_0, loss_1) + suffix)

class Recorder_3:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()
        self.loss0 = tf.keras.metrics.Mean()
        self.loss1 = tf.keras.metrics.Mean()
        self.loss2 = tf.keras.metrics.Mean()
        # self.precision = tf.keras.metrics.Precision()
        # self.recall = tf.keras.metrics.Recall()

        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}, loss_0: {:.4f},loss_1: {:.4f},loss_2: {:.4f}'

    def record(self, losses, loss_0,loss_1,loss_2):
        self.loss.update_state(losses)
        self.loss0.update_state(loss_0)
        self.loss1.update_state(loss_1)
        self.loss2.update_state(loss_2)
        # self.precision.update_state(labels, predictions)
        # self.recall.update_state(labels, predictions)

    def reset(self):
        self.loss.reset_states()
        self.loss0.reset_states()
        self.loss1.reset_states()
        self.loss2.reset_states()
        # self.precision.reset_states()
        # self.recall.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        loss0 = self.loss0.result().numpy()
        loss1 = self.loss1.result().numpy()
        loss2 = self.loss2.result().numpy()
        # precision = self.precision.result().numpy()
        # recall = self.recall.result().numpy()
        # f1 = 2 * precision * recall / (precision + recall + 1e-6)  # avoid division by 0
        return [loss, loss0, loss1, loss2]#,precision, recall, f1]

    def score(self):
        return self._results()[-1].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss,loss0,loss1, loss2 = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss,loss0,loss1,loss2) + suffix)
