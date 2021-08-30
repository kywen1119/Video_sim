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
