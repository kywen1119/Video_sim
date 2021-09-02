"""
    Implementation of the Generalized Pooling Operator (GPO) for aggregating single-modality features.
"""

import tensorflow as tf
import math


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = tf.zeros(shape=[length, d_model])
    position = tf.expand_dims(tf.range(0, length, 1), axis=1)
    div_term = tf.math.exp((tf.range(0, d_model, 2, dtype=tf.float32) *
                            -(math.log(10000.0) / d_model)))

    pe_unstack = tf.unstack(pe, axis=-1)
    sin_values = tf.math.sin(tf.cast(position, tf.float32) * div_term)
    cos_values = tf.math.cos(tf.cast(position, tf.float32) * div_term)
    pe_unstack[0::2] = tf.unstack(sin_values, axis=-1)
    pe_unstack[1::2] = tf.unstack(cos_values, axis=-1)
    pe = tf.stack(pe_unstack, axis=-1)
    return pe


class GPO(tf.keras.layers.Layer):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = tf.keras.layers.GRU(d_hidden, return_sequences=True)
        self.bi_gru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        self.linear = tf.keras.layers.Dense(1, use_bias=False)

    def __call__(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        # chunk the features to remove all zero padding dimensions
        max_len = tf.keras.backend.max(lengths)
        max_len = tf.math.minimum(max_len, features.shape[1])
        features = features[:, :max_len, :]

        pool_weights, mask = self.compute_pool_weights(lengths, max_len)
        
        neg_const = tf.ones(shape=[features.shape[0], max_len, features.shape[-1]])
        neg_const *= -10000
        zero_const = tf.zeros(shape=[features.shape[0], max_len, features.shape[-1]])

        expanded_mask = tf.tile(tf.expand_dims(tf.math.logical_not(mask), -1), [1, 1, features.shape[-1]])
        sorted_features = tf.where(expanded_mask, neg_const, features)
        sorted_features = tf.sort(sorted_features, axis=1, direction='DESCENDING')
        #tf.print('mask', expanded_mask.shape, tf.math.reduce_sum(tf.cast(expanded_mask[0, :, 0], tf.int32), axis=0), tf.math.reduce_sum(tf.cast(expanded_mask[1, :, 0], tf.int32), axis=0))
        sorted_features = tf.where(expanded_mask, zero_const, sorted_features)
        #tf.print('sorted', sorted_features[0])
        #tf.print('pooling coe', pool_weights[0])

        pooled_features = tf.math.reduce_sum(sorted_features * pool_weights, axis=1)
        return pooled_features, pool_weights

    def compute_pool_weights(self, lengths, max_len): 
        pe_max_len = self.get_pe(max_len)
        pes = tf.tile(tf.expand_dims(pe_max_len, axis=0), [lengths.shape[0], 1, 1])
        mask = tf.tile(tf.expand_dims(tf.range(0, max_len, 1), axis=0), [lengths.shape[0], 1])
        mask = (mask < tf.expand_dims(lengths, axis=-1))
        expanded_mask = tf.tile(tf.expand_dims(tf.math.logical_not(mask), -1), [1, 1, pes.shape[-1]])
        #tf.print('mask', expanded_mask.shape, tf.math.reduce_sum(tf.cast(expanded_mask[0, :, 0], tf.int32), axis=0), tf.math.reduce_sum(tf.cast(expanded_mask[1, :, 0], tf.int32), axis=0))
        pes = tf.where(expanded_mask, tf.zeros(shape=[pes.shape[0], max_len, pes.shape[-1]]), pes)

        #tf.print('pes', pes[0], pes[0].shape)
        out_emb = self.bi_gru(pes, mask=mask)
        scores = self.linear(out_emb)

        neg_const = tf.ones(shape=[pes.shape[0], max_len, 1])
        neg_const *= -10000
        expanded_mask = tf.expand_dims(tf.math.logical_not(mask), -1)
        scores = tf.where(expanded_mask, neg_const, scores)
        weights = tf.keras.activations.softmax(scores / 0.1, axis=1)
        return weights, mask

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        if length.ref() in self.pe_database:
            return self.pe_database[length.ref()]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length.ref()] = pe
            return pe


if __name__ == '__main__':
    features = tf.keras.Input(shape=(20, 1024), batch_size=128)
    lengths = tf.keras.Input(shape=(1,), batch_size=128, dtype='int32')
    gpo = GPO(32, 32)
    outputs = gpo(features, lengths)
