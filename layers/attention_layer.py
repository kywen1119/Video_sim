import tensorflow as tf

from layers.utils import create_initializer, get_shape_list, \
  reshape_to_matrix, transpose_for_attention_dot
from layers.algebra_layer import dot_attention_prob, get_activation


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class BasicAttentionLayer(tf.keras.layers.Layer):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`."""

  # pylint: disable=too-many-arguments
  def __init__(self, num_attention_heads=1,
               size_per_head=512,
               query_act=None, key_act=None, value_act=None,
               use_bias=True, attention_probs_dropout_prob=0.0,
               initializer_config=None, do_clip_inf=False, batch_size=None,
               from_seq_length=None, to_seq_length=None, **kwargs):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need".

    Args:
      num_attention_heads (int): Number of attention heads.
      size_per_head (int): Size of each attention head.
      query_act (str): (optional) Activation function for the query transform.
      key_act (str): (optional) Activation function for the key transform.
      value_act (str): (optional) Activation function for the value transform.
      use_bias (bool): Whether the layer uses a bias vector.
      attention_probs_dropout_prob (float): Dropout probability of the attention
        probabilities.
      initializer_config (dict): Optional dense initialization config.
      batch_size (int): Optional when input is 3D. If the input is 2D, this
        might be the batch size of the 3D version of the `from_tensor` and
        `to_tensor`.
      from_seq_length (int): Optional when input is 3D. If the input is 2D,
        this might be the seq length of the 3D version of the `from_tensor`.
      to_seq_length (int): Optional when input is 3D. If the input is 2D, this
        might be the seq length of the 3D version of the `to_tensor`.
      do_clip_inf (bool): Clip inf in `attention_scores`, this op will use more
        memory.

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """
    super(BasicAttentionLayer, self).__init__(**kwargs)

    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.batch_size = batch_size
    self.from_seq_length = from_seq_length
    self.to_seq_length = to_seq_length
    self.do_clip_inf = do_clip_inf

    # `query_layer` = [B*F, N*H]
    self.query_layer = tf.keras.layers.Dense(
      num_attention_heads * size_per_head,
      use_bias=use_bias,
      activation=get_activation(query_act),
      name="query",
      kernel_initializer=create_initializer(initializer_config))

    # `key_layer` = [B*T, N*H]
    self.key_layer = tf.keras.layers.Dense(
      num_attention_heads * size_per_head,
      use_bias=use_bias,
      activation=get_activation(key_act),
      name="key",
      kernel_initializer=create_initializer(initializer_config))

    # `value_layer` = [B*T, N*H]
    self.value_layer = tf.keras.layers.Dense(
      num_attention_heads * size_per_head,
      use_bias=use_bias,
      activation=get_activation(value_act),
      name="value",
      kernel_initializer=create_initializer(initializer_config))

  def call(self, inputs, **kwargs):
    """Do basic attention.

    Args:
      inputs (list): List of two float tensors, [from_tensor, to_tensor].
      **kwargs:

    Returns:
      (tf.Tensor, tf.Tensor): `context_layer` is a float Tensor of shape
        [batch_size, from_seq_length, num_attention_heads * size_per_head]. (If
        `do_return_2d_tensor` is true, this will be of shape
        [batch_size * from_seq_length, num_attention_heads * size_per_head]).
        `attention_probs` is a float Tensor of shape [batch_size,
        num_attention_heads, from_seq_length, to_seq_length].

    """
    from_tensor, to_tensor = inputs
    attention_mask = kwargs.pop("attention_mask", None)

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
      raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

    do_return_2d_tensor = True
    batch_size = self.batch_size
    from_seq_length = self.from_seq_length
    to_seq_length = self.to_seq_length

    if len(from_shape) == 3:
      batch_size = from_shape[0]
      from_seq_length = from_shape[1]
      to_seq_length = to_shape[1]
      do_return_2d_tensor = False
    elif len(from_shape) == 2:
      if (self.batch_size is None) or (self.from_seq_length is None) or (
          self.to_seq_length is None):
        raise ValueError(
          "When passing in rank 2 tensors to BasicAttentionLayer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # `from_tensor_2d` = [batch_size*from_seq_length, from_width]
    from_tensor_2d = reshape_to_matrix(from_tensor)
    # `to_tensor_2d` = [batch_size*to_seq_length, from_width]
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_output` = [B*F, N*H]
    query_output = self.query_layer(from_tensor_2d, **kwargs)

    # `key_output` = [B*T, N*H]
    key_output = self.key_layer(to_tensor_2d, **kwargs)

    # `value_output` = [B*T, N*H]
    value_output = self.value_layer(to_tensor_2d, **kwargs)

    # `query_output` = [B, N, F, H]
    # tf.print(query_output.shape,batch_size)
    query_output = transpose_for_attention_dot(
      query_output, batch_size, self.num_attention_heads,
      from_seq_length, self.size_per_head)

    # `key_output` = [B, N, T, H]
    key_output = transpose_for_attention_dot(
      key_output, batch_size, self.num_attention_heads,
      to_seq_length, self.size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = dot_attention_prob(
      query_output, key_output,
      size_per_head=self.size_per_head,
      attention_mask=attention_mask,
      dropout_prob=self.attention_probs_dropout_prob,
      do_clip_inf=self.do_clip_inf)

    # `value_output` = [B, N, T, H]
    value_output = transpose_for_attention_dot(
      value_output, batch_size, self.num_attention_heads, to_seq_length,
      self.size_per_head)

    # `context_layer` = [B, N, F, H]
    context_output = tf.linalg.matmul(attention_probs, value_output)

    # `context_layer` = [B, F, N, H]
    context_output = tf.transpose(context_output, [0, 2, 1, 3])

    if do_return_2d_tensor:
      # `context_layer` = [B*F, N*H]
      output_shape_0 = -1 if batch_size == -1 else batch_size * from_seq_length

      context_output = tf.reshape(
        context_output,
        [output_shape_0, self.num_attention_heads * self.size_per_head])
    else:
      # `context_layer` = [B, F, N*H]
      context_output = tf.reshape(
        context_output,
        [batch_size, from_seq_length,
         self.num_attention_heads * self.size_per_head])

    return context_output, attention_probs


def basic_attention_layer(from_tensor,  # pylint:disable=too-many-arguments
                          to_tensor, attention_mask=None,
                          num_attention_heads=1, size_per_head=512,
                          query_act=None, key_act=None, value_act=None,
                          use_bias=True, attention_probs_dropout_prob=0.0,
                          initializer_config=None,
                          batch_size=None, from_seq_length=None,
                          to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor (tf.Tensor): float Tensor of shape [batch_size, from_seq_length,
        from_width].
    to_tensor (tf.Tensor): float Tensor of shape [batch_size, to_seq_length,
      to_width].
    attention_mask (tf.Tensor): (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions
      in the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads (int): Number of attention heads.
    size_per_head (int): Size of each attention head.
    query_act (str): (optional) Activation function for the query transform.
    key_act (str): (optional) Activation function for the key transform.
    value_act (str): (optional) Activation function for the value transform.
    use_bias (bool): Whether the layer uses a bias vector.
    attention_probs_dropout_prob (float): Dropout probability of the attention
      probabilities.
    initializer_config (dict): Optional dense initialization config.
    batch_size (int): Optional when input is 3D. If the input is 2D, this might
      be the batch size of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length (int): Optional when input is 3D. If the input is 2D, this
      might be the seq length of the 3D version of the `from_tensor`.
    to_seq_length (int): Optional when input is 3D. If the input is 2D, this
      might be the seq length of the 3D version of the `to_tensor`.

  Returns:
    (tf.Tensor, tf.Tensor): `attention_probs` is a float Tensor of shape
      [batch_size, num_attention_heads, from_seq_length, to_seq_length].
      `context_layer` is a float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is true,
      this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  context_layer, attention_probs = BasicAttentionLayer(
    num_attention_heads=num_attention_heads, size_per_head=size_per_head,
    query_act=query_act, key_act=key_act, value_act=value_act,
    use_bias=use_bias,
    attention_probs_dropout_prob=attention_probs_dropout_prob,
    initializer_config=initializer_config, batch_size=batch_size,
    from_seq_length=from_seq_length, to_seq_length=to_seq_length
  )([from_tensor, to_tensor], attention_mask=attention_mask)

  return context_layer, attention_probs