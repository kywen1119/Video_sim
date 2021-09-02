from absl import logging
import tensorflow as tf

from layers.utils import get_shape_list, reshape_from_matrix, \
    reshape_to_matrix, create_initializer, get_layer_share_mapping, \
    conditional_jit_scope, get_int_shape

from layers.attention_layer import BasicAttentionLayer
from layers.algebra_layer import TransformerFFN, LayerProcess


# from archer.ft.faster_transformer import use_ft_ops, ft_bert_input, \
#   ft_bert_output  # todo not suport


# pylint: disable=too-many-instance-attributes
class TransformerEncoderLayer(tf.keras.layers.Layer):
    """A single Transformer Encoder Layer, support standard FastTransformer."""

    # pylint: disable=too-many-arguments
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 intermediate_size=3072, intermediate_act_fn=None,
                 pre_layer_process_config=None, post_layer_process_config=None,
                 use_bias=True, attention_probs_dropout_prob=0.,
                 act_dropout_prob=0., initializer_config=None,
                 attention_do_clip_inf=False, seq_length=None,
                 output_initializer_config=None, **kwargs):
        """Multi-headed Transformer Encoder Layer from "Attention is All You Need".

        This is almost an exact implementation of the original Transformer encoder.

        Args:
          hidden_size (int): Hidden size of the Transformer.
          num_attention_heads (int): Number of attention heads in the Transformer.
          intermediate_size (int): The size of the "intermediate" (a.k.a., feed
            forward) layer.
          intermediate_act_fn (str or function): The non-linear activation function to
            apply to the output of the intermediate/feed-forward layer.
          pre_layer_process_config (dict): Operation before `dense`, see
            `layer_process` in `algebra_layer.py`.
          post_layer_process_config (dict): Operation after `dense`, see
            `layer_process` in `algebra_layer.py`.
          use_bias (bool): Whether the layer uses a bias vector.
          attention_probs_dropout_prob (float): Dropout probability of the attention
            probabilities.
          act_dropout_prob (float): Dropout probability of the activation
            probabilities in transformer ffn.
          initializer_config (dict): Initialization config, default to `truncated
            normal` with stddev 0.02.
          output_initializer_config (dict): Initialization config of output layer,
            default to `truncated normal` with stddev 0.02.
          attention_do_clip_inf (bool): Clip inf in `attention_scores`, this op
            will use more memory.
          batch_size (int): Optional when input is 3D. If the input is 2D, this
            might be the batch size of the 3D version of the `layer_input`.
          seq_length (int): Optional when input is 3D. If the input is 2D, this
            might be the seq length of the 3D version of the `layer_input`.

        """
        super(TransformerEncoderLayer, self).__init__(**kwargs)

        if pre_layer_process_config is None:
            pre_layer_process_config = {}
        if post_layer_process_config is None:
            post_layer_process_config = {}

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.use_bias = use_bias
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.act_dropout_prob = act_dropout_prob
        self.initializer_config = initializer_config
        self.output_initializer_config = \
            initializer_config if output_initializer_config is None else output_initializer_config
        self.pre_layer_process_config = pre_layer_process_config
        self.post_layer_process_config = post_layer_process_config
        self.attention_do_clip_inf = attention_do_clip_inf

        self.batch_size = kwargs.get("batch_size")
        self.seq_length = seq_length

    def build(self, input_shape):
        """Build TransformerEncoderLayer.

        Args:
          input_shape (tf.TensorShape): A tf.TensorShape object of
            `input_tensor`.

        """
        input_shape = get_shape_list(input_tensor_shape=input_shape,
                                     expected_rank=[2, 3])
        if len(input_shape) == 3:
            self.batch_size = get_int_shape(input_shape[0], self.batch_size, True)
            self.seq_length = get_int_shape(input_shape[1], self.seq_length)
        elif len(input_shape) == 2:
            if (self.batch_size is None) or (self.seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to TransformerEncoderLayer, "
                    "the values for `batch_size` and `seq_length` "
                    "must all be specified.")

        attention_head_size = int(self.hidden_size / self.num_attention_heads)

        self.attention_input_process = LayerProcess(
            sequence=self.pre_layer_process_config.get("sequence", "none"),
            dropout_prob=self.pre_layer_process_config.get("dropout_prob", 0.),
            norm_type=self.pre_layer_process_config.get("norm_type", "none"),
            epsilon=self.pre_layer_process_config.get("epsilon", 1e-6),
            dropout_broadcast_dims=self.pre_layer_process_config.get(
                "dropout_broadcast_dims"),
            force_fp32=self.pre_layer_process_config.get("force_fp32", True),
            name="pre_layer_process"
        )

        self.basic_attention_layer = BasicAttentionLayer(
            num_attention_heads=self.num_attention_heads,
            size_per_head=attention_head_size,
            use_bias=self.use_bias,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_config=self.initializer_config,
            do_clip_inf=self.attention_do_clip_inf,
            batch_size=self.batch_size,
            from_seq_length=self.seq_length,
            to_seq_length=self.seq_length,
            name="self_attention"
        )

        self.attention_output_dense = tf.keras.layers.Dense(
            self.hidden_size,
            name="attention_output_dense",
            kernel_initializer=create_initializer(self.output_initializer_config),
        )

        self.attention_output_process = LayerProcess(
            sequence=self.post_layer_process_config.get("sequence", "none"),
            dropout_prob=self.post_layer_process_config.get("dropout_prob", 0.),
            norm_type=self.post_layer_process_config.get("norm_type", "none"),
            epsilon=self.post_layer_process_config.get("epsilon", 1e-6),
            dropout_broadcast_dims=self.post_layer_process_config.get(
                "dropout_broadcast_dims"),
            force_fp32=self.post_layer_process_config.get("force_fp32", True),
            name="post_layer_process",
        )

        self.transformer_ffn_layer = TransformerFFN(
            self.hidden_size,
            self.intermediate_size,
            self.intermediate_act_fn,
            use_bias=self.use_bias,
            pre_layer_process_config=self.pre_layer_process_config,
            post_layer_process_config=self.post_layer_process_config,
            act_dropout_prob=self.act_dropout_prob,
            initializer_config=self.initializer_config,
            output_initializer_config=self.output_initializer_config,
            name="transformer_ffn",
        )

        self.built = True

    def call(self, inputs, **kwargs):
        """Do the single layer transform.

        Args:
          inputs (tf.Tensor): float Tensor of shape [batch_size, seq_length,
            hidden_size] or [batch_size*seq_length, hidden_size].
          **kwargs:
            attention_mask (tf.Tensor): Optional float Tensor of shape [batch_size,
            seq_length, seq_length], with 1 for positions that can be attended
            to and 0 in positions that should not be.
            use_fast_transformer (bool): Whether to use `fast_transformer` in
              inference.
            ft_index_tensors (list): Optional list of tensors. If `ft_index_tensors`
             is not None, we will use `fast_transformer` with padding remover.
            ft_numpy_params (dict): Dict of numpy params for fast transformer.

        Returns:
          (tf.Tensor, tf.Tensor, tf.Tensor): float Tensors with the same shape as
            inputs.

        """
        layer_input = inputs
        attention_mask = kwargs.pop("attention_mask", None)
        user_fast_transformer = kwargs.pop("use_fast_transformer", False)
        ft_index_tensors = kwargs.pop("ft_index_tensors", [])
        ft_compress_tensor = kwargs.pop("ft_compress_tensor", None)
        ft_numpy_params = kwargs.pop("ft_numpy_params", None)
        attention_input = self.attention_input_process(
            [None, layer_input], **kwargs)

        attention_output, attention_probs = self.basic_attention_layer(
            [attention_input, attention_input],
            attention_mask=attention_mask, **kwargs)

        attention_output = self.attention_output_dense(attention_output, **kwargs)
        attention_output = self.attention_output_process(
            [layer_input, attention_output], **kwargs)

        # The activation is only applied to the "intermediate" hidden layer.
        layer_output = self.transformer_ffn_layer(attention_output, **kwargs)
        ft_layer_output = layer_output
        if user_fast_transformer:  # todo not supported
            logging.error("fast transformer not supported")
            if ft_index_tensors:
                logging.info("Enable Fast Transformer with padding remover.")
                if ft_compress_tensor is None:
                    raise NotImplementedError
                ft_input = ft_compress_tensor
            else:
                logging.info("Enable Fast Transformer without padding remover.")
                ft_input = layer_input
            ft_layer_output = use_ft_ops(ft_input, self.trainable_weights,
                                         self.batch_size, self.seq_length,
                                         attention_mask,
                                         self.hidden_size,
                                         self.num_attention_heads,
                                         ft_index_tensors=ft_index_tensors,
                                         ft_numpy_params=ft_numpy_params)

        return layer_output, attention_probs, ft_layer_output


# pylint:disable=too-many-arguments
def transformer_encoder_layer(layer_input,
                              attention_mask=None,
                              hidden_size=768,
                              num_attention_heads=12,
                              intermediate_size=3072,
                              intermediate_act_fn=None,
                              pre_layer_process_config=None,
                              post_layer_process_config=None,
                              use_bias=True,
                              attention_probs_dropout_prob=0.,
                              act_dropout_prob=0.,
                              initializer_config=None,
                              batch_size=None,
                              seq_length=None,
                              use_fast_transformer=False,
                              ft_index_tensors=None):
    """Multi-headed Transformer Encoder Layer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    Args:
      layer_input (tf.Tensor): float Tensor of shape [batch_size, seq_length,
        hidden_size] or [batch_size*seq_length, hidden_size].
      attention_mask (tf.Tensor): Optional float Tensor of shape [batch_size,
        seq_length, seq_length], with 1 for positions that can be attended
        to and 0 in positions that should not be.
      hidden_size (int): Hidden size of the Transformer.
      num_attention_heads (int): Number of attention heads in the Transformer.
      intermediate_size (int): The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn (str or function): The non-linear activation function to
        apply to the output of the intermediate/feed-forward layer.
      pre_layer_process_config (dict): Operation before `dense`, see
        `layer_process` in `algebra_layer.py`.
      post_layer_process_config (dict): Operation after `dense`, see
        `layer_process` in `algebra_layer.py`.
      use_bias (bool): Whether the layer uses a bias vector.
      attention_probs_dropout_prob (float): Dropout probability of the attention
        probabilities.
      act_dropout_prob (float): Dropout probability of the activation
        probabilities in transformer ffn.
      initializer_config (dict): Initialization config, default to `truncated
        normal` with stddev 0.02.
      batch_size (int): Optional when input is 3D. If the input is 2D, this might
        be the batch size of the 3D version of the `layer_input`.
      seq_length (int): Optional when input is 3D. If the input is 2D, this might
        be the seq length of the 3D version of the `layer_input`.
      use_fast_transformer (bool): Whether to use `fast_transformer` in inference.
      ft_index_tensors (list): Optional list of tensors. If `ft_index_tensors` is
        not None, we will use `fast_transformer` with padding remover.

      Returns:
        (tf.Tensor, tf.Tensor): float Tensors with the same shape as layer_input.

      Raises:
        ValueError: A Tensor shape or parameter is invalid.
    """

    layer_output, attention_probs, ft_layer_output = TransformerEncoderLayer(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        intermediate_act_fn=intermediate_act_fn,
        pre_layer_process_config=pre_layer_process_config,
        post_layer_process_config=post_layer_process_config,
        use_bias=use_bias,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        act_dropout_prob=act_dropout_prob,
        initializer_config=initializer_config,
        batch_size=batch_size,
        seq_length=seq_length
    )(layer_input, attention_mask=attention_mask,
                               use_fast_transformer=use_fast_transformer,
                               ft_index_tensors=ft_index_tensors)

    return layer_output, attention_probs, ft_layer_output


class TransformerEncoder(tf.keras.layers.Layer):
    """Standard Transformer Encoder."""

    # pylint: disable=too-many-arguments
    def __init__(self, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072,
                 intermediate_act_fn=None, pre_layer_process_config=None,
                 post_layer_process_config=None, use_bias=True,
                 hidden_dropout_prob=0., attention_probs_dropout_prob=0.,
                 act_dropout_prob=0., initializer_config=None,
                 layer_groups=None, seq_length=None, attention_do_clip_inf=False,
                 output_initializer_config=None, **kwargs):
        """Multi-headed, multi-layer Transformer from "Attention is All You Need".

        This is almost an exact implementation of the original Transformer encoder.

        See the original paper:
        https://arxiv.org/abs/1706.03762

        Also see:
        https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

        Args:
          hidden_size (int): Hidden size of the Transformer.
          num_hidden_layers (int): Number of layers (blocks) in the Transformer.
          num_attention_heads (int): Number of attention heads in the Transformer.
          intermediate_size (int): The size of the "intermediate" (a.k.a., feed
            forward) layer.
          intermediate_act_fn (str or function): The non-linear activation function to
            apply to the output of the intermediate/feed-forward layer.
          pre_layer_process_config (dict): Operation before `dense`, see
            `layer_process` in `algebra_layer.py`.
          post_layer_process_config (dict): Operation after `dense`, see
            `layer_process` in `algebra_layer.py`.
          use_bias (bool): Whether the layer uses a bias vector.
          hidden_dropout_prob (float): Dropout probability for the hidden layers.
          attention_probs_dropout_prob (float): Dropout probability of the attention
            probabilities.
          act_dropout_prob (float): Dropout probability of the activation
            probabilities in transformer ffn.
          initializer_config (dict): Initialization config, default to `truncated
            normal` with stddev 0.02.
          output_initializer_config (dict): Initialization config of output layer,
            default to `truncated normal` with stddev 0.02.
          layer_groups (list): Share weight between layers in one group.
          batch_size (int): Optional when input is 3D. If the input is 2D, this
            might be the batch size of the 3D version of the `layer_input`.
          seq_length (int): Optional when input is 3D. If the input is 2D, this
            might be the seq length of the 3D version of the `layer_input`.
          attention_do_clip_inf (bool): Clip inf in `attention_scores`, this op
            will use more memory.

        Raises:
          ValueError: A Tensor shape or parameter is invalid.
        """
        super(TransformerEncoder, self).__init__(**kwargs)

        if pre_layer_process_config is None:
            # BERT is none, NMT is n
            pre_layer_process_config = {"sequence": "none", "epsilon": 1e-6,
                                        "dropout_prob": hidden_dropout_prob,
                                        "norm_type": "layer"}
            logging.warning("Use BERT default pre_layer_process_config: {}.".format(
                pre_layer_process_config))

        if post_layer_process_config is None:
            # BERT is dan, NMT is da
            post_layer_process_config = {"sequence": "dan", "epsilon": 1e-6,
                                         "dropout_prob": hidden_dropout_prob,
                                         "norm_type": "layer"}
            logging.warning("Use BERT default post_layer_process_config: {}.".format(
                post_layer_process_config))

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_act_fn = intermediate_act_fn
        self.pre_layer_process_config = pre_layer_process_config
        self.post_layer_process_config = post_layer_process_config
        self.use_bias = use_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.act_dropout_prob = act_dropout_prob
        self.initializer_config = initializer_config
        layer_groups = layer_groups if layer_groups else []
        self.layer_groups = [list(map(str, lg)) for lg in layer_groups]
        self.output_initializer_config = \
            initializer_config if output_initializer_config is None else output_initializer_config
        self.batch_size = kwargs.get("batch_size")
        self.seq_length = seq_length
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.attention_do_clip_inf = attention_do_clip_inf
        self.encoder_layers = {}

    def build(self, input_shape):
        """Build TransformerEncoder.

        Args:
          input_shape (tf.TensorShape): A tf.TensorShape object of
            `input_tensor`.

        """
        input_shape = get_shape_list(input_tensor_shape=input_shape,
                                     expected_rank=[2, 3])
        if len(input_shape) == 3:
            self.batch_size = get_int_shape(input_shape[0], self.batch_size, True)
            self.seq_length = get_int_shape(input_shape[1], self.seq_length)
        elif len(input_shape) == 2:
            if (self.batch_size is None) or (self.seq_length is None):
                raise ValueError(
                    "When passing in rank 2 tensors to TransformerEncoderLayer, "
                    "the values for `batch_size` and `seq_length` "
                    "must all be specified.")
        input_width = input_shape[-1]

        # The Transformer performs sum residuals on all layers so the input needs
        # to be the same as the hidden size.
        if input_width != self.hidden_size:
            raise ValueError(
                "The width of the input tensor ({}) != hidden size ({})".format(
                    input_width, self.hidden_size))

        # get share mapping
        share_layer_mapping = {}
        if self.layer_groups is not None:
            share_layer_mapping = get_layer_share_mapping(self.layer_groups)
        logging.info("Share layer mapping: {}.".format(share_layer_mapping))

        for layer_idx in range(self.num_hidden_layers):
            layer_idx = str(layer_idx)
            if layer_idx in share_layer_mapping:
                real_layer_idx = share_layer_mapping[layer_idx]
                logging.info("Share weight between layer {} and layer {}.".format(
                    layer_idx, real_layer_idx))
                self.encoder_layers[layer_idx] = self.encoder_layers[real_layer_idx]
            else:
                self.encoder_layers[layer_idx] = TransformerEncoderLayer(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    intermediate_size=self.intermediate_size,
                    intermediate_act_fn=self.intermediate_act_fn,
                    pre_layer_process_config=self.pre_layer_process_config,
                    post_layer_process_config=self.post_layer_process_config,
                    use_bias=self.use_bias,
                    attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                    act_dropout_prob=self.act_dropout_prob,
                    initializer_config=self.initializer_config,
                    output_initializer_config=self.output_initializer_config,
                    batch_size=self.batch_size,
                    seq_length=self.seq_length,
                    attention_do_clip_inf=self.attention_do_clip_inf,
                    name="layer_{}".format(layer_idx),
                )

        self.built = True

    def call(self, inputs, **kwargs):
        """Do transformer encode.

        Args:
          inputs (tf.Tensor): float Tensor of shape [batch_size, seq_length,
            hidden_size].
          **kwargs:
            attention_mask (tf.Tensor): Optional int32 Tensor of shape [batch_size,
              seq_length, seq_length], with 1 for positions that can be attended to
              and 0 in positions that should not be.
            use_fast_transformer (bool): Whether to use `fast_transformer` in
              inference.
            ft_numpy_params (dict): Dict of numpy params for fast transformer.
            input_mask (tf.Tensor): A int32 Tensor. Whether to use `fast_transformer
             rm padding` in inference, work with `use_fast_transformer` param.

        Returns:
          (tf.Tensor or list, list): A (list of) float tensor(s) of shape
            [batch_size, seq_length, hidden_size], the final hidden layer of the
            Transformer.
            all_attention_probs: list of float tensors.

        """
        input_tensor = inputs
        attention_mask = kwargs.pop("attention_mask", None)
        use_fast_transformer = kwargs.pop("use_fast_transformer", False)
        ft_numpy_params = kwargs.pop("ft_numpy_params", None)
        input_mask = kwargs.pop("input_mask", None)

        # run ft to rm padding if input_mask provided
        ft_index_tensors = []
        ft_compress_tensor = None
        if use_fast_transformer:  # todo fast transformer not supported
            logging.error("fast transformer not supported")
            if input_mask is not None:
                ft_compress_tensors = ft_bert_input(input_tensor, input_mask,
                                                    self.batch_size, self.seq_length,
                                                    self.num_attention_heads,
                                                    self.attention_head_size)
                ft_compress_tensor = ft_compress_tensors[0]
                ft_index_tensors = ft_compress_tensors[1]
            else:
                logging.info("use faster_transformer without rm padding")

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = reshape_to_matrix(input_tensor)

        all_layer_outputs = []
        all_attention_probs = []
        for layer_idx in range(self.num_hidden_layers):
            # want to know more XLA and JIT, you can see:
            # https://www.tensorflow.org/xla and
            # https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html
            with conditional_jit_scope():
                layer_output, attention_probs, ft_layer_output = self.encoder_layers[
                    str(layer_idx)](prev_output, attention_mask=attention_mask,
                                    use_fast_transformer=use_fast_transformer,
                                    ft_compress_tensor=ft_compress_tensor,
                                    ft_index_tensors=ft_index_tensors,
                                    ft_numpy_params=ft_numpy_params, **kwargs)
                if ft_compress_tensor is not None:
                    ft_compress_tensor = ft_layer_output
                prev_output = layer_output
                # reshape and restore `layer_output` to ordinary `layer_output`
                layer_output = reshape_from_matrix(
                    layer_output, [self.batch_size, self.seq_length, self.hidden_size])
                if ft_index_tensors:
                    layer_output = ft_bert_output(ft_layer_output, ft_index_tensors,
                                                  self.batch_size, self.seq_length,
                                                  self.num_attention_heads,
                                                  self.attention_head_size)
                all_layer_outputs.append(layer_output)
                all_attention_probs.append(attention_probs)

        return all_layer_outputs, all_attention_probs


# pylint:disable=R0913,R0914
def transformer_encoder(input_tensor,
                        attention_mask=None,
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        intermediate_act_fn=None,
                        pre_layer_process_config=None,
                        post_layer_process_config=None,
                        use_bias=True,
                        hidden_dropout_prob=0.,
                        attention_probs_dropout_prob=0.,
                        act_dropout_prob=0.,
                        initializer_config=None,
                        layer_groups=None,
                        use_fast_transformer=False,
                        input_mask=None):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor (tf.Tensor): float Tensor of shape [batch_size, seq_length,
        hidden_size].
      attention_mask (tf.Tensor): Optional int32 Tensor of shape [batch_size,
        seq_length, seq_length], with 1 for positions that can be attended
        to and 0 in positions that should not be.
      hidden_size (int): Hidden size of the Transformer.
      num_hidden_layers (int): Number of layers (blocks) in the Transformer.
      num_attention_heads (int): Number of attention heads in the Transformer.
      intermediate_size (int): The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn (str or function): The non-linear activation function to
        apply to the output of the intermediate/feed-forward layer.
      pre_layer_process_config (dict): Operation before `dense`, see
        `layer_process` in `algebra_layer.py`.
      post_layer_process_config (dict): Operation after `dense`, see
        `layer_process` in `algebra_layer.py`.
      use_bias (bool): Whether the layer uses a bias vector.
      hidden_dropout_prob (float): Dropout probability for the hidden layers.
      attention_probs_dropout_prob (float): Dropout probability of the attention
        probabilities.
      act_dropout_prob (float): Dropout probability of the activation
        probabilities in transformer ffn.
      initializer_config (dict): Initialization config, default to `truncated
        normal` with stddev 0.02.
      layer_groups (list): Share weight between layers in one group.
      use_fast_transformer (bool): Whether to use `fast_transformer` in inference.
      input_mask (tf.Tensor): A int32 Tensor. Whether to use `fast_transformer rm
        padding` in inference, work with `use_fast_transformer` param.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.

    """
    final_outputs, _ = TransformerEncoder(
        hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        intermediate_act_fn=intermediate_act_fn,
        pre_layer_process_config=pre_layer_process_config,
        post_layer_process_config=post_layer_process_config, use_bias=use_bias,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        act_dropout_prob=act_dropout_prob, initializer_config=initializer_config,
        layer_groups=layer_groups)(input_tensor, attention_mask=attention_mask,
                                   use_fast_transformer=use_fast_transformer,
                                   input_mask=input_mask)

    return final_outputs
