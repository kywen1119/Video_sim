import tensorflow as tf
import numpy as np

def get_angles(pos, i, d_model):
    # 获得sin内部的值 (pos / 10000 ^ 2i/d_model)
    # i索引是embedding维度的索引
    # pos是seq_len维度的位置
    # 这里传进来的i不是i本身 而是2i或者2i+1
    # 所以下面做了一个 (2 * (i//2)) 操作
    # 为了就是得到公式里的2i值
    angle_rate = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rate

def positional_encoding(position, d_model):
    # 为了形成 [seq_len, d_model]矩阵 需要增加维度
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # angle_rads shape: (seq_len, d_model) 其中seq_len就是position

    # apply sin to even indices in the array; 2i
    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # 0::2 - start pos: 0, step:2

    # apply cos to odd indices in the array; 2i+1
    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# 构造 Q K V 矩阵的计算
# 点积注意力被缩小了深度的平方根倍。
# 对于较大的深度值，点积的大小会增大，从而推动 softmax 函数往仅有很小的梯度的方向靠拢，导致hard softmax。
# 遮挡（mask）与 -1e9（接近于负无穷）相乘。
# 遮挡与缩放的 Q 和 K 的矩阵乘积相加，并在 softmax 之前立即应用。
# 目标是将这些单元归零，因为 softmax 的较大负数输入在输出中接近于零。
# 其实最后就是找到了 seq_len_q 个 最匹配的 depth_v 维度的向量
def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
        output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    # 首先计算 Q K 矩阵相乘 获得的矩阵为(seq_len, seq_len)
    matmul_qk = tf.matmul(q, k, transpose_b=True) # (batch_size, seq_len_q, seq_len_k)

    # scale matmul_qk
    # dk is the dim of q and k
    # 缩放 matmul_qk ---> sqrt(d_model)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    # -1e9的mask在softmax之后趋近0
    if mask is not None:
        scaled_attention_logits += ((1.0-mask) * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1
    # 当 softmax 在 K 上进行归一化后，它的值决定了分配到 Q 的重要程度。
    # softmax 在最后一个轴（seq_len_k）上归一化
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# 每个多头注意力块有三个输入：Q（请求）、K（主键）、V（数值）。这些输入经过线性（Dense）层，并分拆成多头。
# 将上面定义的 scaled_dot_product_attention 函数应用于每个头（进行了广播（broadcasted）以提高效率）。
# 注意力这步必须使用一个恰当的 mask。
# 然后将每个头的注意力输出连接起来（用tf.transpose 和 tf.reshape），并放入最后的 Dense 层。
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # 保证 d_model 可以整除 拆分成 num_heads 个多头
        assert d_model % self.num_heads == 0

        # 拆分后的维度
        self.depth = d_model // self.num_heads

        # Input 经过三个线性变换称为 Q K V
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # move num_heads to dim 2
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # multi head 在 concat 以后再过一个全连接层
        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)

        return output, attention_weights

MIDDLE_DROPOUT_RATE = 0.0
class NonLocalAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(NonLocalAttention, self).__init__()
        self.d_model = d_model

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.relu = tf.keras.layers.ReLU()
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(MIDDLE_DROPOUT_RATE)

    def call(self, in_query, in_memory):
        q = self.wq(in_query)  # (batch_size, seq_len, d_model)
        k = self.wk(in_memory)  # (batch_size, seq_len, d_model)
        v = self.wv(in_memory)  # (batch_size, seq_len, d_model)

        # output.shape == (batch_size, seq_len_q, d_model)
        # attention_weights.shape == (batch_size, seq_len_q, seq_len_k)
        output, attention_weights = scaled_dot_product_attention(q, k, v)

        # (batch_size, seq_len_q, d_model)
        output = self.layernorm(output)
        output = self.relu(output)
        output = self.dense(output)
        output = self.dropout(output)

        return output

class NonLocalLayer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model):
        super(NonLocalLayer, self).__init__()
        self.nl_layers = [
            NonLocalAttention(d_model) for _ in range(num_layers)
        ]

    def call(self, in_query, in_memory):
        for i in range(len(self.nl_layers)):
            nl_out = self.nl_layers[i](in_query, in_memory)
            nl_out = nl_out + in_query
            in_query = nl_out

        return nl_out


# 全连接层 FFN
# 由两层全联接层组成，两层之间有一个 ReLU 激活函数。
# 原始论文中 d_model:512  dff:2048
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'), # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
    ])


# 一个 EncoderLayer层包括
# Multi-Head Attention 和 FeedForward
# 在 Multi-Head Attention 和 FeedForward最后的Dense后部加入dropout
# 以及其中的residual和add norm
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        """
        Args:
            d_model: 特征维度
            num_heads: 多头数量
            dff: 全连接层维度
            rate: Dropout值
        """
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

        return out2


# 整体结构
# 输入transformer encoder的是一个 (batch_size, seq_len, d_model)维度的特征
# 这里不需要Embedding层因为我们输入的不是word而已经是一个embedding
class Transformer_Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, dropout_rate=0.1):
        super(Transformer_Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 位置信息 与sequence长度和embedding维度有关
        self.pos_encoding = positional_encoding(seq_len, self.d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

    def call(self, x, mask=None):
        # 第一步 input_embedding + positional encoding
        # batch中的每个instance 相同位置要加上的pos_encoding一致
        # 所以 x.shape==(bs, seq_len, d_model), pos_encoding.shape==(1, seq_len, d_model)
        x += self.pos_encoding
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x # (bs, input_seq_len , d_model)


class Video_transformer(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers=1, output_size=1024, seq_len=32, dropout=0.2):
        super().__init__()
        # self.fc = tf.keras.layers.Dense(output_size, activation='relu')

        self.frame_tf_encoder = Transformer_Encoder(num_layers=num_hidden_layers,
                                                   d_model=output_size,
                                                   num_heads=4,
                                                   dff=output_size * 4,
                                                   seq_len=seq_len)
        self.num_heads = 4

    def call(self, inputs, **kwargs):
        image_embeddings, mask = inputs
        _, num_segments, _ = image_embeddings.shape
        if mask is not None:  # in case num of images is less than num_segments
            image_mask = tf.sequence_mask(mask, maxlen=num_segments) # b,32
            images_mask = tf.cast(tf.expand_dims(image_mask, -1), tf.float32) # b,32,1
            image_embeddings = tf.multiply(image_embeddings, images_mask)
        i_mask = tf.expand_dims(image_mask, 1) # b,1,32
        i_mask = tf.expand_dims(i_mask, 1) # b,1,1,32
        attention_mask = tf.cast(tf.tile(i_mask, [1,self.num_heads,num_segments,1]), tf.float32)
        # image_embeddings = self.fc(image_embeddings)
        # image_embeddings += self.pos_encoding
        x = self.frame_tf_encoder(image_embeddings, mask=attention_mask)
        return x, 1.0-images_mask