import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Model
from transformers import TFBertModel, create_optimizer
# from layers.transformer_layer import TransformerEncoder
from model_transformer import Transformer_Encoder


class NeXtVLAD(tf.keras.layers.Layer):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

        self.new_feature_size = expansion * feature_size // groups
        self.expand_dense = tf.keras.layers.Dense(self.expansion * self.feature_size)
        # for group attention
        self.attention_dense = tf.keras.layers.Dense(self.groups, activation=tf.nn.sigmoid)
        # self.activation_bn = tf.keras.layers.BatchNormalization()

        # for cluster weights
        self.cluster_dense1 = tf.keras.layers.Dense(self.groups * self.cluster_size, activation=None, use_bias=False)
        # self.cluster_dense2 = tf.keras.layers.Dense(self.cluster_size, activation=None, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(rate=dropout, seed=1)
        self.fc = tf.keras.layers.Dense(output_size, activation=None)

    def build(self, input_shape):
        self.cluster_weights2 = self.add_weight(name="cluster_weights2",
                                                shape=(1, self.new_feature_size, self.cluster_size),
                                                initializer=tf.keras.initializers.glorot_normal, trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        image_embeddings, mask = inputs
        _, num_segments, _ = image_embeddings.shape
        if mask is not None:  # in case num of images is less than num_segments
            images_mask = tf.sequence_mask(mask, maxlen=num_segments)
            images_mask = tf.cast(tf.expand_dims(images_mask, -1), tf.float32)
            image_embeddings = tf.multiply(image_embeddings, images_mask)
        inputs = self.expand_dense(image_embeddings)
        attention = self.attention_dense(inputs)

        attention = tf.reshape(attention, [-1, num_segments * self.groups, 1])
        reshaped_input = tf.reshape(inputs, [-1, self.expansion * self.feature_size])

        activation = self.cluster_dense1(reshaped_input)
        # activation = self.activation_bn(activation)
        activation = tf.reshape(activation, [-1, num_segments * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)  # shape: batch_size * (max_frame*groups) * cluster_size
        activation = tf.multiply(activation, attention)  # shape: batch_size * (max_frame*groups) * cluster_size

        a_sum = tf.reduce_sum(activation, -2, keepdims=True)  # shape: batch_size * 1 * cluster_size
        a = tf.multiply(a_sum, self.cluster_weights2)  # shape: batch_size * new_feature_size * cluster_size
        activation = tf.transpose(activation, perm=[0, 2, 1])  # shape: batch_size * cluster_size * (max_frame*groups)

        reshaped_input = tf.reshape(inputs, [-1, num_segments * self.groups, self.new_feature_size])

        vlad = tf.matmul(activation, reshaped_input)  # shape: batch_size * cluster_size * new_feature_size
        vlad = tf.transpose(vlad, perm=[0, 2, 1])  # shape: batch_size * new_feature_size * cluster_size
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * self.new_feature_size])

        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


# def get_angles(pos, i, d_model):
#     # 获得sin内部的值 (pos / 10000 ^ 2i/d_model)
#     # i索引是embedding维度的索引
#     # pos是seq_len维度的位置
#     # 这里传进来的i不是i本身 而是2i或者2i+1
#     # 所以下面做了一个 (2 * (i//2)) 操作
#     # 为了就是得到公式里的2i值
#     angle_rate = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
#     return pos * angle_rate

# def positional_encoding(position, d_model):
#     # 为了形成 [seq_len, d_model]矩阵 需要增加维度
#     angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                             np.arange(d_model)[np.newaxis, :],
#                             d_model)
#     # angle_rads shape: (seq_len, d_model) 其中seq_len就是position

#     # apply sin to even indices in the array; 2i
#     # 将 sin 应用于数组中的偶数索引（indices）；2i
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # 0::2 - start pos: 0, step:2

#     # apply cos to odd indices in the array; 2i+1
#     # 将 cos 应用于数组中的奇数索引；2i+1
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

#     pos_encoding = angle_rads[np.newaxis, ...]

#     return tf.cast(pos_encoding, dtype=tf.float32)


class Video_transformer(tf.keras.layers.Layer):
    def __init__(self, num_hidden_layers=1, output_size=1024, seq_len=32, dropout=0.2):
        super().__init__()
        self.fc = tf.keras.layers.Dense(output_size, activation='relu')
        # self.frame_tf_encoder = TransformerEncoder(hidden_size=output_size, num_hidden_layers=num_hidden_layers,
        #          num_attention_heads=8, intermediate_size=3072)
        self.frame_tf_encoder = Transformer_Encoder(num_layers=num_hidden_layers,
                                                   d_model=output_size,
                                                   num_heads=4,
                                                   dff=output_size * 4,
                                                   seq_len=seq_len)
        self.num_heads = 4
        # self.pos_encoding = positional_encoding(seq_len, output_size)

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
        image_embeddings = self.fc(image_embeddings)
        # image_embeddings += self.pos_encoding
        x = self.frame_tf_encoder(image_embeddings, mask=attention_mask)
        return x, 1.0-images_mask


class Multimodal_transformer(tf.keras.layers.Layer):
    def __init__(self, config, num_hidden_layers=1, output_size=1024, seq_len=32, dropout=0.2):
        super().__init__()
        self.frame_fc = tf.keras.layers.Dense(output_size, activation='relu')
        self.bert_fc = tf.keras.layers.Dense(output_size, activation='relu')
        # self.frame_tf_encoder = TransformerEncoder(hidden_size=output_size, num_hidden_layers=num_hidden_layers,
        #          num_attention_heads=8, intermediate_size=3072)
        self.video_transformer = Video_transformer(num_hidden_layers=1, output_size=config.vlad_hidden_size, 
                                                    seq_len=config.max_frames, dropout=config.dropout)
        self.tf_encoder = Transformer_Encoder(num_layers=num_hidden_layers,
                                                   d_model=output_size,
                                                   num_heads=4,
                                                   dff=output_size * 4,
                                                   seq_len=seq_len)
        self.num_heads = 4
        # self.pos_encoding = positional_encoding(seq_len, output_size)

    def call(self, inputs_frame, inputs_seq, **kwargs):
        image_embeddings, frame_mask = inputs_frame
        image_embeddings = self.frame_fc(image_embeddings)
        video_tf_embedding, _ = self.video_transformer([image_embeddings, frame_mask])
        text_embeddings, text_mask = inputs_seq # bert feature
        _, num_segments, _ = image_embeddings.shape
        _, num_words, _ = text_embeddings.shape
        if frame_mask is not None:  # in case num of images is less than num_segments
            image_mask = tf.sequence_mask(frame_mask, maxlen=num_segments) # b,32
            images_mask = tf.cast(tf.expand_dims(image_mask, -1), tf.float32) # b,32,1
            i_mask = tf.expand_dims(image_mask, 1) # b,1,32
            i_mask = tf.cast(tf.expand_dims(i_mask, 1), tf.float32)  # b,1,1,32
        if text_mask is not None:
            texts_mask = tf.cast(tf.expand_dims(text_mask, -1), tf.float32) # b,32,1
            t_mask = tf.expand_dims(text_mask, 1) # b,1,32
            t_mask = tf.cast(tf.expand_dims(t_mask, 1), tf.float32) # b,1,1,32
        mask = tf.concat([i_mask,t_mask], -1)
        attention_mask = tf.cast(tf.tile(mask, [1,self.num_heads,num_segments+num_words,1]), tf.float32)
        
        text_embeddings = self.bert_fc(text_embeddings)
        embeddings = tf.concat([video_tf_embedding, text_embeddings], 1)
        # image_embeddings += self.pos_encoding
        x = self.tf_encoder(embeddings, mask=attention_mask)
        return x, 1.0-tf.concat([images_mask, texts_mask], 1), tf.reduce_max(video_tf_embedding, axis=1), tf.reduce_max(text_embeddings, axis=1)



class SENet(tf.keras.layers.Layer):
    def __init__(self, channels, ratio=8, **kwargs):
        super(SENet, self).__init__(**kwargs)
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False),
            tf.keras.layers.Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        ])

    def call(self, inputs, **kwargs):
        se = self.fc(inputs)
        outputs = tf.math.multiply(inputs, se)
        return outputs


class ConcatDenseSE(tf.keras.layers.Layer):
    """Fusion using Concate + Dense + SENet"""

    def __init__(self, hidden_size, se_ratio, **kwargs):
        super().__init__(**kwargs)
        self.fusion = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fusion_dropout = tf.keras.layers.Dropout(0.2)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def call(self, inputs, **kwargs):
        embeddings = tf.concat(inputs, axis=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding


class MultiModal(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel.from_pretrained(config.bert_dir)
        # self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
        #                          output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.video_transformer = Video_transformer(num_hidden_layers=1, output_size=config.vlad_hidden_size, 
                                                    seq_len=config.max_frames, dropout=config.dropout)
        self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')
        self.bert_map = tf.keras.layers.Dense(1024, activation ='relu')

        self.bert_optimizer, self.bert_lr = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables, self.num_bert, self.normal_variables, self.all_variables = None, None, None, None

    def call(self, inputs, **kwargs):
        bert_embedding = self.bert([inputs['input_ids'], inputs['mask']])[1] # 0:last_hidden_state,1:poller_output
        bert_embedding = self.bert_map(bert_embedding)
        frame_num = tf.reshape(inputs['num_frames'], [-1])
        vision_embedding, images_mask = self.video_transformer([inputs['frames'], frame_num])
        # super_neg = images_mask * -10000 # b, 32, 1
        vision_embedding = tf.reduce_max(vision_embedding, axis=1)
        vision_embedding = vision_embedding * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32) # avoid videos which don't have frame features
        final_embedding = self.fusion([vision_embedding, bert_embedding])
        predictions = self.classifier(final_embedding)

        return predictions, final_embedding, vision_embedding, bert_embedding

    def get_variables(self):
        if not self.all_variables:  # is None, not initialized
            self.bert_variables = self.bert.trainable_variables
            self.num_bert = len(self.bert_variables)
            self.normal_variables = self.video_transformer.trainable_variables + self.fusion.trainable_variables + \
                                    self.classifier.trainable_variables + self.bert_map.trainable_variables # 这个之前忘记加了
            self.all_variables = self.bert_variables + self.normal_variables
        return self.all_variables

    def optimize(self, gradients):
        bert_gradients = gradients[:self.num_bert]
        self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
        normal_gradients = gradients[self.num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))


class MultiModal_JT(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel.from_pretrained(config.bert_dir)
        # self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
        #                          output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.transformer = Multimodal_transformer(config, num_hidden_layers=1, output_size=config.vlad_hidden_size, 
                                                    seq_len=config.max_frames+config.bert_seq_length, dropout=config.dropout)
        # self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')
        # self.bert_map = tf.keras.layers.Dense(1024, activation ='relu')

        self.bert_optimizer, self.bert_lr = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables, self.num_bert, self.normal_variables, self.all_variables = None, None, None, None

    def call(self, inputs, **kwargs):
        bert_embedding = self.bert([inputs['input_ids'], inputs['mask']])[0] # 0:last_hidden_state,1:poller_output
        # bert_embedding = self.bert_map(bert_embedding)
        frame_num = tf.reshape(inputs['num_frames'], [-1])
        video_embedding, mask, image_emb, text_emb = self.transformer([inputs['frames'], frame_num], [bert_embedding, inputs['mask']])
        super_neg = mask * -10000 # b, 32, 1
        video_embedding = tf.reduce_max(video_embedding + super_neg, axis=1)
        # video_embedding = video_embedding * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32) # avoid videos which don't have frame features
        # final_embedding = self.fusion([vision_embedding, bert_embedding])
        predictions = self.classifier(video_embedding)

        return predictions, video_embedding, image_emb, text_emb

    def get_variables(self):
        if not self.all_variables:  # is None, not initialized
            self.bert_variables = self.bert.trainable_variables
            self.num_bert = len(self.bert_variables)
            self.normal_variables = self.transformer.trainable_variables + self.classifier.trainable_variables# self.fusion.trainable_variables + \
                                    # + self.bert_map.trainable_variables # 这个之前忘记加了
            self.all_variables = self.bert_variables + self.normal_variables
        return self.all_variables

    def optimize(self, gradients):
        bert_gradients = gradients[:self.num_bert]
        self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
        normal_gradients = gradients[self.num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))
