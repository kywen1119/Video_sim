import logging
import tensorflow as tf


class Recorder:
    def __init__(self):
        self.loss = tf.keras.metrics.Mean()

        self.pattern = 'Epoch: {}, step: {}, loss: {:.4f}'

    def record(self, losses):
        self.loss.update_state(losses)

    def reset(self):
        self.loss.reset_states()

    def _results(self):
        loss = self.loss.result().numpy()
        return [loss]

    def score(self):
        return self._results()[-1].numpy()

    def log(self, epoch, num_step, prefix='', suffix=''):
        loss = self._results()
        logging.info(prefix + self.pattern.format(epoch, num_step, loss) + suffix)
