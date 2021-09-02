"""
Simple math computation. It contains transformer encoder components
include LayerProcess used to do pre-process or post-process on a layer,
TransformerFFNN, Biaslayer, normalizaiotn layer and some basic function
such as get_activation.
"""
import math

import numpy
import tensorflow as tf
from absl import logging

from utils import get_shape_list, create_initializer, clip_inf, \
    gather_indexes
# from archer import compat  todo make show it's not needed


def gelu(input_tensor, approximate=True):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor (tf.Tensor): float Tensor to perform activation.
      approximate (bool): Use tanh approximation

    Returns:
      tf.Tensor: `input_tensor` with the GELU activation applied.
    """
    if approximate:
        # noqa:on
        cdf = 0.5 * (1.0 + tf.math.tanh((numpy.sqrt(2 / numpy.pi) * (
                input_tensor + 0.044715 * tf.math.pow(input_tensor, 3)))))
        # noqa:off
    else:
        # https://github.com/tensorflow/tensorflow/issues/25052
        # fp16 should be ready
        cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / numpy.sqrt(2.0)))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
      activation_string (str): String name of the activation function.

    Returns:
      function: A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return
        `activation_string`.

    Raises:
      ValueError: The `activation_string` does not correspond to a known
        activation.
    """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string, str):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()

    if act == "linear":
        act_func = None
    elif act == "relu":
        act_func = tf.nn.relu
    elif act == "gelu":
        act_func = gelu
    elif act == "tanh":
        act_func = tf.math.tanh
    elif act == "swish":
        act_func = tf.nn.swish
    else:
        raise ValueError("Unsupported activation: {}".format(act))

    return act_func


def dropout_with_broadcast_dims(input_tensor, dropout_prob=0.,
                                broadcast_dims=None, **kwargs):
    """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

    Instead of specifying noise_shape, this function takes broadcast_dims -
    a list of dimension numbers in which noise_shape should be 1.  The random
    keep/drop tensor has dimensionality 1 along these dimensions.

    Args:
      input_tensor (tf.Tensor): A floating point tensor.
      dropout_prob (float): The probability that each element is dropped.
      broadcast_dims (list): An optional list of integers the dimensions along
        which to broadcast the keep/drop flags.
      **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".

    Returns:
      tf.Tensor: Tensor of the same shape as x.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    assert "noise_shape" not in kwargs
    if broadcast_dims:
        shape = get_shape_list(input_tensor)
        ndims = len(shape)
        # Allow dimensions like "-1" as well.
        broadcast_dims = [dim + ndims if dim < 0 else dim for dim in
                          broadcast_dims]
        kwargs["noise_shape"] = [
            1 if i in broadcast_dims else shape[i] for i in range(ndims)]
    if kwargs.get("training") is not None:
        # logging.warning("Training kwargs are not supported by dropout now.")
        kwargs.pop("training")
    return tf.nn.dropout(input_tensor, rate=dropout_prob, **kwargs)


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor (tf.Tensor): float Tensor.
      dropout_prob (float): The probability of dropping out a value (
        NOT of *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
        tf.Tensor: A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


class ZeroAdd(tf.keras.layers.Layer):
    """Resnet connection with zero initialization.

      Another type of resnet connection which returns
       `previous_value + gamma * x`.
      gamma is a trainable scalar and initialized with zero. It is useful when a
      module is plugged into a trained model and we want to make sure it matches
      the original model's performance.

    """

    def build(self, input_shape):  # pylint: disable=unused-argument
        """Add `gamma` without using input_shape.

        Args:
          input_shape (tuple): ignore

        """
        self.gamma = self.add_weight("gamma", shape=(),
                                     initializer=tf.zeros_initializer(),
                                     dtype=self.dtype)
        self.built = True

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        """Do previous_value + gamma * x.

        Args:
          inputs (list): List of two tensors, [previous_value, input_tensor].

        Returns:
          tf.Tensor: previous_value + gamma * input_tensor.
        """
        return inputs[0] + self.gamma * inputs[1]


def zero_add(previous_value, input_tensor):
    """Resnet connection with zero initialization.

    Another type of resnet connection which returns previous_value + gamma * x.
    gamma is a trainable scalar and initialized with zero. It is useful when a
    module is plugged into a trained model and we want to make sure it matches
    the original model's performance.

    Args:
      previous_value (tf.Tensor):  A tensor.
      input_tensor (tf.Tensor): A tensor.

    Returns:
      tf.Tensor: previous_value + gamma * input_tensor.
    """
    output_tensor = ZeroAdd()([previous_value, input_tensor])
    return output_tensor


# pylint: disable=too-few-public-methods
class ApplyNorm(tf.keras.layers.Layer):
    """Apply various norm.
    We need to reuse variables, so we write this keras layer.

    """

    def __init__(self, norm_type, epsilon=0.001, axis=-1, force_fp32=True,
                 **kwargs):
        """Apply Normalization with keras Layer.

          Args:
            norm_type (str): Current support 'layer', 'batch', 'none'.
            epsilon (float): Stable computation.
            axis (int): The axis that should be normalized (typically
              the features axis).
            force_fp32 (bool): force cast `input_tensor` to fp32.

          Returns:
            tf.Tensor: A Tensor after `norm` transform.

          Raises:
            ValueError: If norm_type not in 'layer', 'batch', 'none'.

          """
        super(ApplyNorm, self).__init__(**kwargs)
        if norm_type == "layer":
            self.norm_layer = tf.keras.layers.LayerNormalization(
                axis=axis, epsilon=epsilon, name="layer_normalization")
        elif norm_type == "batch":
            # training is passed by **kwargs, TODO: need test
            self.norm_layer = tf.keras.layers.BatchNormalization(
                axis=axis, epsilon=epsilon, name="batch_normalization")
        elif norm_type == "none":
            self.norm_layer = None
        else:
            raise ValueError("norm_type must be one of: 'layer', 'batch', 'none'.")

        self.force_fp32 = force_fp32

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        """Apply normalization to inputs.

        Args:
          inputs (tf.Tensor): Tensor to be normalized.
          **kwargs:

        Returns:

        """
        if self.norm_layer is None:
            return inputs

        tensor_for_norm = inputs
        if self.force_fp32 and inputs.dtype.base_dtype == tf.float16:
            # logging.warning('Force cast dtype float16 to float32 in layer norm.')
            tensor_for_norm = tf.cast(inputs, tf.float32)

        tensor_for_output = self.norm_layer(tensor_for_norm, **kwargs)

        if self.force_fp32 and inputs.dtype.base_dtype == tf.float16:
            tensor_for_output = tf.cast(tensor_for_output, tf.float16)

        return tensor_for_output


def apply_norm(input_tensor, norm_type, epsilon=0.001, axis=-1,
               force_fp32=True):
    """Apply Normalization.

    Args:
      input_tensor (tf.Tensor): Tensor to be normalized.
      norm_type (str): Current support 'layer', 'batch', 'none'.
      epsilon (float): Stable computation.
      axis (int): The axis that should be normalized (typically
        the features axis).
      force_fp32 (bool): force cast `input_tensor` to fp32.

    Returns:
      tf.Tensor: A Tensor after `norm` transform.

    Raises:
      ValueError: If norm_type not in 'layer', 'batch', 'none'.

    """
    tensor_for_output = ApplyNorm(
        norm_type, epsilon=epsilon, axis=axis, force_fp32=force_fp32)(input_tensor)

    return tensor_for_output


# pylint: disable=too-few-public-methods
class LayerProcess(tf.keras.layers.Layer):
    """Apply a sequence of functions to the input or output of a layer."""

    def __init__(self, sequence="none", dropout_prob=0., norm_type="none",
                 epsilon=1e-6, dropout_broadcast_dims=None, **kwargs):
        """Apply a sequence of functions to the input or output of a layer.

        The sequence in `layer_process_config` is specified as a string which may
        contain the following characters:
          a: add previous_value
          n: apply normalization
          d: apply dropout
          z: zero add
        For example, if sequence=="dna", then the output is
          previous_value + normalize(dropout(x))

        Args:
          sequence (str): The sequence is specified as a string, see above.
          dropout_prob (float): Parameter for dropout.
          norm_type (str): See apply_norm().
          epsilon (float): Parameter for normalization.
          dropout_broadcast_dims (list): An optional list of integers specifying
            in which dimensions to broadcast the dropout decisions to save memory.

        """
        force_fp32 = kwargs.pop("force_fp32", True)
        super(LayerProcess, self).__init__(**kwargs)
        self.sequence = sequence
        self.dropout_prob = dropout_prob
        self.dropout_broadcast_dims = dropout_broadcast_dims
        self.zero_add_layer = None
        self.norm_layer = None
        if self.sequence != "none":
            if "z" in self.sequence:
                # only do zero add once
                assert self.sequence.count("z") == 1
                self.zero_add_layer = ZeroAdd(name="zero_add")
            if "n" in self.sequence:
                # only do norm once
                assert self.sequence.count("n") == 1
                self.norm_layer = ApplyNorm(
                    norm_type, epsilon=epsilon, name="apply_norm", force_fp32=force_fp32)

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        """Do `sequence` transform.

        Args:
          inputs (list): List of two tensors, [previous_value, input_tensor].

        Returns:
          tf.Tensor: A Tensor after `sequence` transform.
        """
        previous_value, input_tensor = inputs
        if "a" in self.sequence:
            assert previous_value is not None

        if self.sequence == "none":
            return input_tensor

        for operation in self.sequence:
            if operation == "a":
                input_tensor += previous_value
            elif operation == "z":
                input_tensor = self.zero_add_layer([previous_value, input_tensor],
                                                   **kwargs)
            elif operation == "n":
                input_tensor = self.norm_layer(input_tensor, **kwargs)
            else:
                assert operation == "d", "Unknown sequence step {}".format(operation)
                input_tensor = dropout_with_broadcast_dims(
                    input_tensor, self.dropout_prob,
                    broadcast_dims=self.dropout_broadcast_dims, **kwargs)

        return input_tensor


def layer_process(previous_value, input_tensor,
                  sequence="none", dropout_prob=0., norm_type="none",
                  epsilon=1e-6, dropout_broadcast_dims=None):
    """Apply a sequence of functions to the input or output of a layer.

    The sequence in `layer_process_config` is specified as a string which may
    contain the following characters:
      a: add previous_value
      n: apply normalization
      d: apply dropout
      z: zero add
    For example, if sequence=="dna", then the output is
      previous_value + normalize(dropout(x))

    Args:
      previous_value (tf.Tensor): A Tensor, to be added as a residual
        connection ('a').
      input_tensor (tf.Tensor): A Tensor to be transformed.
      sequence (str): The sequence is specified as a string, see above.
      dropout_prob (float): Parameter for dropout.
      norm_type (str): See apply_norm().
      epsilon (float): Parameter for normalization.
      dropout_broadcast_dims (list): An optional list of integers specifying
        in which dimensions to broadcast the dropout decisions to save memory.
    Returns:
      tf.Tensor: A Tensor after `sequence` transform.
    """
    # noqa:on
    output_tensor = LayerProcess(
        sequence=sequence, dropout_prob=dropout_prob, norm_type=norm_type,
        epsilon=epsilon, dropout_broadcast_dims=dropout_broadcast_dims)(
        [previous_value, input_tensor])
    # noqa:off
    return output_tensor


def dot_attention_prob(query, key,
                       size_per_head,
                       attention_mask=None,
                       dropout_prob=0.,
                       do_clip_inf=False):
    """Dot attention in BERT and other transformer variants.

    Args:
      query (tf.Tensor): Float Tensor of shape [batch_size, num_attention_heads,
        query_length, size_per_head].
      key (tf.Tensor): Float Tensor of shape [batch_size, num_attention_heads,
        key_length, size_per_head].
      size_per_head (int): Size of each attention head.
      attention_mask (tf.Tensor): (optional) int32 Tensor of shape [batch_size,
        query_length, key_length]. The values should be 1 or 0. 1 is useful.
        The attention scores will effectively be set to -infinity for any
        positions in the mask that are 0, and will be unchanged for positions
        that are 1.
      dropout_prob (float): Dropout probability of the attention probabilities.
      do_clip_inf (bool): Clip inf in `attention_scores`, this op will use more
        memory.

    Returns:
      tf.Tensor: Probability tensor of shape [batch_size, num_attention_heads,
        query_length, key_length].

    """
    # [batch_size, num_attention_heads, query_length, key_length]
    # scale first
    query = tf.math.multiply(query, 1.0 / math.sqrt(float(size_per_head)))
    attention_scores = tf.linalg.matmul(query, key, transpose_b=True)

    if attention_mask is not None:
        # `attention_mask` = [batch_size, 1, query_length, key_length]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and
        # 0.0 for masked positions, this operation will create a tensor
        # which is 0.0 for positions we want to attend and -10000.0 for
        # masked positions.
        adder = (1.0 - tf.dtypes.cast(attention_mask,
                                      attention_scores.dtype)) * -10000.0

        # Since we are adding it to the raw scores before the softmax,
        # this is effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [batch_size, num_attention_heads, query_length,
    # key_length]
    # clip inf to prevent nan
    if do_clip_inf:
        logging.warning("Enable clip inf in dot_attention.")
        attention_scores = clip_inf(attention_scores)
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, dropout_prob)
    return attention_probs


# pylint: disable=too-few-public-methods
class TransformerFFN(tf.keras.layers.Layer):
    """ Feed forward net for transformer. """

    def __init__(self, hidden_size, intermediate_size,
                 intermediate_act_fn, use_bias=True,
                 pre_layer_process_config=None,
                 post_layer_process_config=None,
                 act_dropout_prob=0.,
                 initializer_config=None,
                 output_initializer_config=None, **kwargs):
        super(TransformerFFN, self).__init__(**kwargs)

        self.act_dropout_prob = act_dropout_prob

        self.pre_layer_process = LayerProcess(
            sequence=pre_layer_process_config.get("sequence", "none"),
            dropout_prob=pre_layer_process_config.get("dropout_prob", 0.),
            norm_type=pre_layer_process_config.get("norm_type", "none"),
            epsilon=pre_layer_process_config.get("epsilon", 1e-6),
            dropout_broadcast_dims=pre_layer_process_config.get(
                "dropout_broadcast_dims"),
            force_fp32=pre_layer_process_config.get("force_fp32", True),
            name="pre_layer_process")

        self.intermediate_output_dense = tf.keras.layers.Dense(
            intermediate_size,
            name="intermediate_dense",
            use_bias=use_bias,
            activation=get_activation(intermediate_act_fn),
            kernel_initializer=create_initializer(initializer_config))

        if output_initializer_config is None:
            output_initializer_config = initializer_config
        self.layer_output_dense = tf.keras.layers.Dense(
            hidden_size,
            name="output_dense",
            use_bias=use_bias,
            kernel_initializer=create_initializer(output_initializer_config))

        self.post_layer_process = LayerProcess(
            sequence=post_layer_process_config.get("sequence", "none"),
            dropout_prob=post_layer_process_config.get("dropout_prob", 0.),
            norm_type=post_layer_process_config.get("norm_type", "none"),
            epsilon=post_layer_process_config.get("epsilon", 1e-6),
            dropout_broadcast_dims=post_layer_process_config.get(
                "dropout_broadcast_dims"),
            force_fp32=post_layer_process_config.get("force_fp32", True),
            name="post_layer_process")

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        """Feed forward net for inputs.

        Args:
          inputs (tf.Tensor): A float tensor after attention process.

        Returns:
          tf.Tensor: A float tensor with shape, [batch_size, ..., hidden_size].
        """
        attention_output = inputs
        pre_input = self.pre_layer_process([None, attention_output], **kwargs)
        intermediate_output = self.intermediate_output_dense(pre_input, **kwargs)
        intermediate_output = dropout_with_broadcast_dims(
            intermediate_output, self.act_dropout_prob, **kwargs)
        layer_output = self.layer_output_dense(intermediate_output, **kwargs)
        layer_output = self.post_layer_process([attention_output, layer_output],
                                               **kwargs)
        return layer_output


def transformer_ffn(attention_output,
                    hidden_size,
                    intermediate_size,
                    intermediate_act_fn,
                    use_bias=True,
                    pre_layer_process_config=None,
                    post_layer_process_config=None,
                    act_dropout_prob=0.,
                    initializer_config=None,
                    output_initializer_config=None):
    """ Feed forward net for transformer.

    Args:
      attention_output (tf.Tensor): A float tensor after attention process.
      hidden_size (int): Output size after FFN.
      intermediate_size (int): hidden_size -> intermediate_size -> hidden_size
      intermediate_act_fn (str): Activation function for the hidden transform.
      use_bias (bool): Whether the layer uses a bias vector.
      pre_layer_process_config (dict): Layer process before FFN, see default
        below.
      post_layer_process_config: (dict): Layer process after FFN, see default
        below.
      act_dropout_prob (float): Dropout probability after the Activation.
      initializer_config (dict): Optional dense initialization config.
      output_initializer_config (dict): Optional output initialization config.

    Returns:
      tf.Tensor: A float tensor with shape, [batch_size, ..., hidden_size].

    """
    # The activation is only applied to the "intermediate" hidden layer.
    output_tensor = TransformerFFN(
        hidden_size, intermediate_size,
        intermediate_act_fn, use_bias=use_bias,
        pre_layer_process_config=pre_layer_process_config,
        post_layer_process_config=post_layer_process_config,
        act_dropout_prob=act_dropout_prob,
        initializer_config=initializer_config,
        output_initializer_config=output_initializer_config)(attention_output)
    return output_tensor


class BiasLayer(tf.keras.layers.Layer):
    """Bias only layer."""

    def __init__(self, hidden_size, **kwargs):
        """Create bias with hidden_size."""
        super(BiasLayer, self).__init__(**kwargs)
        self.hidden_size = hidden_size

    def build(self, input_shape):  # pylint: disable=unused-argument
        """Add `bias` without using input_shape.

        Args:
          input_shape (tuple): ignore

        """
        self.bias = self.add_weight("bias",
                                    shape=[self.hidden_size],
                                    initializer=tf.zeros_initializer())
        self.built = True

    def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
        """Add bias to inputs.

        Args:
          inputs (tf.Tensor): A float tensor.

        Returns:
          tf.Tensor: inputs + bias
        """
        return tf.nn.bias_add(inputs, self.bias)


# pylint: disable=too-few-public-methods
class LogitsLayer(tf.keras.layers.Layer):
    """Convert hidden to logits."""

    def __init__(self, embedding_size, vocab_size, hidden_act,
                 initializer_config=None, **kwargs):
        """Create logits layer like BERT."""
        super(LogitsLayer, self).__init__(**kwargs)
        self.output_dense = tf.keras.layers.Dense(
            units=embedding_size,
            activation=get_activation(hidden_act),
            kernel_initializer=create_initializer(initializer_config),
            name="dense")
        self.output_norm = ApplyNorm("layer", epsilon=1e-5, name="apply_norm")
        self.output_bias = BiasLayer(hidden_size=vocab_size, name="output_bias")

    def call(self, inputs, **kwargs):
        """Feed forward inputs to logits.

        Args:
          inputs (tf.Tensor): A float tensor from encoder.

        Returns:
          tf.Tensor: A float tensor with shape, [batch_size, ..., vocab_size].
        """
        embedding_table = kwargs["embedding_table"]
        masked_lm_positions = kwargs.pop("masked_lm_positions", None)
        mlm_output_tensor = inputs
        if masked_lm_positions is not None:
            # gather mlm sequence_output
            mlm_output_tensor = gather_indexes(inputs, masked_lm_positions)

        # dense -> norm -> logits
        # [batch_size * max_predictions_per_seq, embedding_size]
        mlm_output_tensor = self.output_norm(self.output_dense(mlm_output_tensor))
        mlm_logits = tf.matmul(mlm_output_tensor, embedding_table, transpose_b=True)
        # [batch_size * max_predictions_per_seq, vocab_size]
        mlm_logits = self.output_bias(mlm_logits)
        return mlm_logits
