"""
Utils that do not involve computation.

"""
import os
import copy
import contextlib
from functools import wraps

from absl import logging

import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.eager import context


def tf_decorator_helper(tf_decorator):
  """Keep origin func meta information when using tf decorators.

  Args:
    tf_decorator: Any decorator function not wrapped by functools.wraps.

  Returns:
    Wrapped decorator keep all origin meta information.
  """

  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      return tf_decorator(func)(*args, **kwargs)

    return wrapper

  return decorator


def assert_rank(input_tensor_shape, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    input_tensor_shape (tf.TensorShape): A tf.TensorShape object of
      `input_tensor`.
    expected_rank (int or list): Python integer or list of integers,
      expected rank.
    name (str): Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  expected_rank_dict = {}
  if isinstance(expected_rank, int):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:  # pylint: disable=invalid-name
      expected_rank_dict[x] = True

  actual_rank = len(input_tensor_shape)
  if actual_rank not in expected_rank_dict:
    raise ValueError(
      "For the tensor `{}`, the actual rank "
      "`{}` (shape = {}) is not equal to the expected rank `{}`".format(
        name, actual_rank, input_tensor_shape, expected_rank))


# pylint: disable=not-callable
def create_initializer(initializer_config=None):
  """Creates a `tf.initializers` with the given initializer_config.

  Args:
    initializer_config (dict): should provide `name` and `initializers` related
      keyword args.

  Returns:
    tf.keras.initializers.Initializer: initializers in `tf.initializers`
      namespace.
  """
  if initializer_config is None:
    initializer_config = {"name": "truncated_normal",
                          "mean": 0.,
                          "stddev": 0.02}
  initializer_config = copy.deepcopy(initializer_config)
  initializer_name = initializer_config.pop("name")
  try:
    initializer = getattr(tf.initializers, initializer_name)(
      **initializer_config)
  except AttributeError:
    # use tf 2.0
    initializer_name = "".join(map(str.capitalize, initializer_name.split("_")))
    initializer = getattr(tf.initializers, initializer_name)(
      **initializer_config)
  return initializer


@tf_decorator_helper(function.Defun(
  python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
  shape_func=lambda op: [op.inputs[0].get_shape()]))
def convert_gradient_to_tensor(input_tensor):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    input_tensor (tf.Tensor): A `Tensor`.
  Returns:
    tf.Tensor: The input `Tensor`.
  """
  return input_tensor


# pylint: disable=too-many-branches
def get_int_shape(shape_a, shape_b, is_batch_size=False):
  """

  Args:
    shape_a (int): Optional.
    shape_b (int): Optional.
    is_batch_size (bool): If is batch_size, return -1 is both None.

  Returns:
    int:
  """
  int_shape_a = None
  int_shape_b = None
  if shape_a is not None:
    if isinstance(shape_a, int):
      int_shape_a = shape_a
    if isinstance(shape_a, tf.compat.v1.Dimension):
      int_shape_a = shape_a.value
    if isinstance(shape_a, tf.Tensor):
      int_shape_a = shape_a
  if shape_b is not None:
    if isinstance(shape_b, int):
      int_shape_b = shape_b
    if isinstance(shape_b, tf.compat.v1.Dimension):
      int_shape_b = shape_b.value
    if isinstance(shape_b, tf.Tensor):
      int_shape_b = shape_b

  if int_shape_a is None and int_shape_b is None:
    if is_batch_size:
      return -1
    raise ValueError("Both shapes are None.")

  if isinstance(int_shape_a, int) and isinstance(int_shape_b, int):
    if int_shape_a != -1 and int_shape_b != -1:
      assert int_shape_a == int_shape_b
      return int_shape_a
    return -1

  if isinstance(int_shape_a, int):
    return int_shape_a

  return int_shape_b


# pylint: disable=too-many-branches
def get_shape_list(input_tensor=None, input_tensor_shape=None,
                   expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    input_tensor (tf.Tensor): A tf.Tensor object to find the shape of.
    input_tensor_shape (tf.TensorShape): A tf.TensorShape object of
      `input_tensor`.
    expected_rank (int or list): The optional expected rank of `tensor`.
      If this is specified and the `tensor` has a different rank, and
      exception will be thrown.
    name (str): Optional name of the tensor for the error message.

  Returns:
    list: A list of dimensions of the shape of tensor. All static dimensions
      will be returned as python integers, and dynamic dimensions will be
      returned as tf.Tensor scalars.
  """
  if input_tensor is not None:
    if name is None:
      try:
        name = input_tensor.name
      except AttributeError:
        name = "eager_tensor"

  if input_tensor_shape is not None:
    shape = input_tensor_shape
  else:
    shape = input_tensor.shape

  if expected_rank is not None:
    assert_rank(shape, expected_rank, name)
  if not isinstance(shape, list):
    shape = shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  if input_tensor is not None:
    dyn_shape = tf.shape(input_tensor)
  else:
    dyn_shape = input_tensor_shape
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix).

  Args:
    input_tensor (tf.Tensor): rank must >= 2.

  Returns:
    tf.Tensor: output_matrix with rank 2.

  Raises:
    ValueError: If rank of input_tensor is less than 2.

  """
  ndims = len(input_tensor.shape)
  if ndims < 2:
    raise ValueError(
      "Input tensor must have at least rank 2. Shape = {}".format(
        input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = get_shape_list(input_tensor)[-1]
  output_matrix = tf.reshape(input_tensor, [-1, width])
  return output_matrix


def reshape_from_matrix(input_matrix, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor.

  Args:
    input_matrix (tf.Tensor): reshape the input_matrix to tensor.
    orig_shape_list (list): support statical and dynamical shape.

  Returns:
    tf.Tensor: reshaped tensor.

  """
  if len(orig_shape_list) == 2:
    return input_matrix

  output_shape = get_shape_list(input_matrix)

  # TODO(@zhaoshenjian): check use orig_shape_list directly.
  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(input_matrix, orig_dims + [width])


def transpose_for_attention_dot(input_tensor, batch_size, num_attention_heads,
                                seq_length, size_per_head):
  """Transpose a `input_tensor` for attention scoring and dot.

  Args:
    input_tensor (tf.Tensor): float Tensor of shape [batch_size,
      seq_length, num_attention_heads * width].
    batch_size (int): See `input_tensor`, may be optional.
    num_attention_heads (int): Number of attention heads.
    seq_length (int): See `input_tensor`, may be optional.
    size_per_head (int):  Size of each attention head.

  Returns:
    tf.Tensor: A float tensor of shape `[batch_size, num_attention_heads,
      seq_length, width]`.
  """
  output_tensor = tf.reshape(
    input_tensor, [batch_size, seq_length, num_attention_heads, size_per_head])

  output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
  return output_tensor


def create_attention_mask_from_input_mask(from_tensor, to_mask,
                                          mask_type="bidi"):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    mask_type (str): one of (`bidi`,`l2r`,`r2l`), which means bi-directional,
                     left-to-right, and right-to-left respectively.

  Returns:
    tf.Tensor: float Tensor of shape [batch_size, from_seq_length,
      to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.dtypes.cast(
    tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  # bidi: bi-direction attention
  # l2r: left to right attention
  # r2l: right to left attention
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  if mask_type == 'bidi':
    broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
  elif mask_type == 'l2r':
    i = tf.range(from_seq_length)[:, None]
    j = tf.range(to_seq_length)
    broadcast_ones = tf.cast(i >= j - to_seq_length + from_seq_length,
                             dtype=tf.float32)
  elif mask_type == 'r2l':  # TODO: check correctness
    i = tf.range(from_seq_length)[:, None]
    j = tf.range(to_seq_length)
    broadcast_ones = tf.cast(i <= j - to_seq_length + from_seq_length,
                             dtype=tf.float32)
  else:
    raise NotImplementedError

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def clip_inf(input_tensor):
  """clip inf to max or min like tf.saturate_cast.

  Args:
    input_tensor (tf.Tensor): A dtype tensor that may overflow.

  Returns:
    tf.Tensor: new tensor without inf.

  """
  return tf.clip_by_value(input_tensor, input_tensor.dtype.min,
                          input_tensor.dtype.max)


def get_layer_share_mapping(group_list):
  """Get share layer to origin layer mapping, for example,
  group_list = [[2,4,3], [1,5], [0]]
  share_layer_to_origin_layer_mapping = {
    3: 2,
    4: 2,
    5: 1,
  }

  Args:
    group_list (list): list of list of layer numbers.

  Returns:

  """
  share_layer_to_origin_layer_mapping = {}
  for group in group_list:
    if len(group) > 1:
      group = sorted(group)
      for layer_idx in group[1:]:
        share_layer_to_origin_layer_mapping[layer_idx] = group[0]
  return share_layer_to_origin_layer_mapping


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch.

  Args:
    sequence_tensor (tf.Tensor): A float tensor with shape [batch_size,
      max_seq_length, hidden_size], represents the outputs of encoder.
    positions (tf.Tensor): An int32 tensor with shape [batch_size,
      max_predictions_per_seq]. The corresponding positions in `sequence_tensor`
      will be gathered.

  Returns:
    tf.Tensor: A float tensor of shape [batch_size * max_predictions_per_seq,
      hidden_size], gathered hidden states from `sequence_tensor`.
  """
  sequence_shape = get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
    tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  # TODO: check gather_nd
  # flat may speed up gather
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def scatter_update(sequence, updates, positions):
  """Scatter-update a sequence.

  Args:
    sequence (tf.Tensor): A [batch_size, seq_len] or [batch_size, seq_len,
      depth] tensor
    updates (tf.Tensor): A tensor of size batch_size*seq_len(*depth)
    positions (tf.Tensor): A [batch_size, n_positions] tensor

  Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
    [batch_size, seq_len, depth] tensor of "sequence" with elements at
    "positions" replaced by the values at "updates." Updates to index 0 are
    ignored. If there are duplicated positions the update is only applied once.
    Second is a [batch_size, seq_len] mask tensor of which inputs were updated.

  """
  shape = get_shape_list(sequence, expected_rank=[2, 3])
  depth_dimension = (len(shape) == 3)
  if depth_dimension:
    batch_size, length, hidden_size = shape
  else:
    batch_size, length = shape
    hidden_size = 1
    sequence = tf.expand_dims(sequence, -1)
  n_pos = get_shape_list(positions)[1]

  shift = tf.expand_dims(length * tf.range(batch_size), -1)
  flat_positions = tf.reshape(positions + shift, [-1, 1])
  flat_updates = tf.reshape(updates, [-1, hidden_size])
  # updates[flat_positions] = flat_updates
  updates = tf.scatter_nd(flat_positions, flat_updates,
                          [batch_size * length, hidden_size])
  updates = tf.reshape(updates, [batch_size, length, hidden_size])

  flat_updates_mask = tf.ones([batch_size * n_pos], tf.int32)
  # updates_mask[flat_positions] = 1
  updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask,
                               [batch_size * length])
  updates_mask = tf.reshape(updates_mask, [batch_size, length])
  not_first_token = tf.concat([tf.zeros((batch_size, 1), tf.int32),
                               tf.ones((batch_size, length - 1), tf.int32)], -1)
  updates_mask *= not_first_token
  updates_mask_3d = tf.expand_dims(updates_mask, -1)

  # account for duplicate positions
  if sequence.dtype == tf.float32:
    updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
    updates /= tf.maximum(1.0, updates_mask_3d)
  else:
    assert sequence.dtype == tf.int32
    updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
  updates_mask = tf.minimum(updates_mask, 1)
  updates_mask_3d = tf.minimum(updates_mask_3d, 1)

  updated_sequence = (((1 - updates_mask_3d) * sequence) +
                      (updates_mask_3d * updates))
  if not depth_dimension:
    updated_sequence = tf.squeeze(updated_sequence, -1)

  return updated_sequence, updates_mask


def sample_from_softmax(logits, one_hot=False, disallow=None):
  """Sample from softmax by gumbel.

  Args:
    logits (tf.Tensor): [batch_size, ..., hidden_size]
    one_hot (bool): return onehot vector or idx.
    disallow (tf.Tensor): Optional tensor.

  Returns:
    tf.Tensor: one hot [batch_size, ..., hidden_size] or
      position [batch_size, ...].
  """
  if disallow is not None:
    logits -= 1000.0 * disallow
  uniform_noise = tf.random.uniform(get_shape_list(logits), minval=0, maxval=1)
  # pylint: disable=invalid-unary-operand-type
  gumbel_noise = -tf.math.log(-tf.math.log(uniform_noise + 1e-9) + 1e-9)
  sampled_idx = tf.argmax(logits + gumbel_noise, -1, output_type=tf.int32)
  if not one_hot:
    return sampled_idx

  return tf.one_hot(sampled_idx, logits.shape[-1])


# pylint: disable=missing-function-docstring
@contextlib.contextmanager
def dummy_scope():
  yield


def conditional_jit_scope(use_conditional_jit=False,
                          compile_ops=True, separate_compiled_gradients=True):
  """Create jit scope conditionally.

    If ENABLE_CONDITIONAL_JIT is set, we will use conditional_jit.

  Args:
    use_conditional_jit (bool): If true, use conditional_jit.
    compile_ops (bool): enable xla or not.
    separate_compiled_gradients (bool): If true put each gradient subgraph into
      a separate compilation scope.

  Returns:
    scope

  """
  if os.getenv("ENABLE_CONDITIONAL_JIT") == "1":
    use_conditional_jit = True
  if os.getenv("DISABLE_SEPARATE_COMPILED_GRADIENTS"):
    separate_compiled_gradients = False

  if use_conditional_jit and (not context.executing_eagerly()):
    logging.warning("Enable conditional jit ...")
    if hasattr(tf, "xla"):
      return tf.xla.experimental.jit_scope(
        compile_ops=compile_ops,
        separate_compiled_gradients=separate_compiled_gradients)

    # tf <= 1.13
    logging.warning("Please upgrade tf to 1.15.")
    return tf.contrib.compiler.jit.experimental_jit_scope(
      compile_ops=compile_ops,
      separate_compiled_gradients=separate_compiled_gradients)

  return dummy_scope()
