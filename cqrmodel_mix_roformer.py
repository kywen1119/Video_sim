import tensorflow as tf
from tensorflow.python.keras.models import Model
from transformers import TFRoFormerModel, create_optimizer
# from transformers.models.bert.modeling_tf_bert import TFBertMLMHead
# from layers.transformer_layer import TransformerEncoder


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
        self.activation_bn = tf.keras.layers.BatchNormalization() # modify bn

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
        activation = self.activation_bn(activation) # modify bn
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


class MultiModal_mix(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFRoFormerModel.from_pretrained(config.bert_dir)
        self.bert_map = tf.keras.layers.Dense(1024, activation ='relu')
        self.nextvlad_1 = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.fusion_1 = ConcatDenseSE(config.hidden_size, config.se_ratio)

        self.nextvlad_2 = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.fusion_2 = ConcatDenseSE(config.hidden_size, config.se_ratio)

        self.nextvlad_3 = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.fusion_3 = ConcatDenseSE(config.hidden_size, config.se_ratio)

        self.num_labels = config.num_labels
        # batch, num_labels   before sigmoid
        self.classifier_1 = tf.keras.layers.Dense(self.num_labels)#, activation='sigmoid')
        self.classifier_2 = tf.keras.layers.Dense(self.num_labels)#, activation='sigmoid')
        self.classifier_3 = tf.keras.layers.Dense(self.num_labels)#, activation='sigmoid')
        # 原文用frame+audio的特征在dim1求mean
        self.mix_weights = tf.keras.layers.Dense(3)
        self.bn = tf.keras.layers.BatchNormalization()
        self.cl_temperature = 2.0
        self.cl_lambda = 1.0
        
        self.bert_optimizer, self.bert_lr = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables, self.num_bert, self.normal_variables, self.all_variables = None, None, None, None

    def call(self, inputs, **kwargs):
        bert_embedding = self.bert([inputs['input_ids'], inputs['mask']])[0]
        bert_embedding = tf.reduce_max(bert_embedding, 1)
        bert_embedding = self.bert_map(bert_embedding)
        frame_num = tf.reshape(inputs['num_frames'], [-1])
        # frt_mean
        frt_mean = tf.concat([tf.reduce_mean(inputs['frames'], axis=1),bert_embedding], axis=1) 
        frt_mean = self.bn(frt_mean)
        mix_weights = self.mix_weights(frt_mean) # b,3
        mix_weights = tf.nn.softmax(mix_weights, axis=-1)
        # 1
        vision_embedding_1 = self.nextvlad_1([inputs['frames'], frame_num])
        vision_embedding_1 = vision_embedding_1 * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
        final_embedding_1 = self.fusion_1([vision_embedding_1, bert_embedding])
        logits_1 = self.classifier_1(final_embedding_1)
        predictions_1 = tf.nn.sigmoid(logits_1)
        # 2
        vision_embedding_2 = self.nextvlad_2([inputs['frames'], frame_num])
        vision_embedding_2 = vision_embedding_2 * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
        final_embedding_2 = self.fusion_2([vision_embedding_2, bert_embedding])
        logits_2 = self.classifier_2(final_embedding_2)
        predictions_2 = tf.nn.sigmoid(logits_2)
        # 3
        vision_embedding_3 = self.nextvlad_3([inputs['frames'], frame_num])
        vision_embedding_3 = vision_embedding_3 * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
        final_embedding_3 = self.fusion_3([vision_embedding_3, bert_embedding])
        logits_3 = self.classifier_3(final_embedding_3)
        predictions_3 = tf.nn.sigmoid(logits_3)
        # mix
        aux_preds = [predictions_1, predictions_2, predictions_3]
        logits = [logits_1, logits_2, logits_3]
        logits = tf.stack(logits, axis=1)
        embeddings = [final_embedding_1, final_embedding_2, final_embedding_3]
        embeddings = tf.stack(embeddings, axis=1)
        # vision
        # vision_embedding = [vision_embedding_1, vision_embedding_2, vision_embedding_3]
        # vision_embedding = tf.stack(vision_embedding, axis=1)
        vision_embedding = [vision_embedding_1, vision_embedding_2, vision_embedding_3]
        # vision_embedding = tf.stack(vision_embedding, axis=1)
        # mix_vision_embedding = tf.reduce_sum(tf.multiply(tf.expand_dims(mix_weights, -1), vision_embedding), axis=1)
        mix_logit = tf.reduce_sum(tf.multiply(tf.expand_dims(mix_weights, -1), logits), axis=1)
        mix_embedding = tf.reduce_sum(tf.multiply(tf.expand_dims(mix_weights, -1), embeddings), axis=1)
        pred = tf.nn.sigmoid(mix_logit)
        # kl loss
        rank_pred = tf.expand_dims(tf.nn.softmax(mix_logit/self.cl_temperature, axis=-1), axis=1)
        aux_rank_preds = tf.nn.softmax((logits/self.cl_temperature), axis=-1)
        epsilon = 1e-8
        kl_loss = tf.reduce_sum(rank_pred * (tf.math.log(rank_pred + epsilon) - tf.math.log(aux_rank_preds + epsilon)),
                                axis=-1)

        regularization_loss = self.cl_lambda * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1), axis=-1)
        return pred, aux_preds, regularization_loss, mix_embedding, vision_embedding, bert_embedding

    def get_variables(self):
        if not self.all_variables:  # is None, not initialized
            self.bert_variables = self.bert.trainable_variables
            self.num_bert = len(self.bert_variables)
            self.normal_variables = self.nextvlad_1.trainable_variables + self.fusion_1.trainable_variables + \
                                    self.classifier_1.trainable_variables + self.bert_map.trainable_variables + \
                                    self.mix_weights.trainable_variables + self.bn.trainable_variables + \
                                    self.nextvlad_2.trainable_variables + self.fusion_2.trainable_variables + \
                                    self.classifier_2.trainable_variables + self.nextvlad_3.trainable_variables + \
                                    self.fusion_3.trainable_variables + self.classifier_3.trainable_variables
            self.all_variables = self.bert_variables + self.normal_variables
        return self.all_variables

    def optimize(self, gradients):
        bert_gradients = gradients[:self.num_bert]
        self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
        normal_gradients = gradients[self.num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))


class MultiModal_mix2(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel.from_pretrained(config.bert_dir)
        self.nextvlad_1 = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.nextvlad_2 = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.nextvlad_3 = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
                                 output_size=config.vlad_hidden_size, dropout=config.dropout)
        self.mix_weights = tf.keras.layers.Dense(3)
        self.bn = tf.keras.layers.BatchNormalization()

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
        bert_embedding = self.bert([inputs['input_ids'], inputs['mask']])[1]
        bert_embedding = self.bert_map(bert_embedding)
        # frt_mean
        frt_mean = tf.reduce_mean(inputs['frames'], axis=1)
        frt_mean = self.bn(frt_mean)
        mix_weights = self.mix_weights(frt_mean) # b,3
        mix_weights = tf.nn.softmax(mix_weights, axis=-1)
        # 3 nextvlad -> weighted add
        frame_num = tf.reshape(inputs['num_frames'], [-1])
        # 1
        vision_embedding_1 = self.nextvlad_1([inputs['frames'], frame_num])
        vision_embedding_1 = vision_embedding_1 * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
        # 2
        vision_embedding_2 = self.nextvlad_2([inputs['frames'], frame_num])
        vision_embedding_2 = vision_embedding_2 * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
        # 3
        vision_embedding_3 = self.nextvlad_3([inputs['frames'], frame_num])
        vision_embedding_3 = vision_embedding_3 * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
        # mix frame feature
        vision_embedding = [vision_embedding_1, vision_embedding_2, vision_embedding_3]
        vision_embedding = tf.stack(vision_embedding, axis=1)
        mix_vision_embedding = tf.reduce_sum(tf.multiply(tf.expand_dims(mix_weights, -1), vision_embedding), axis=1)

        final_embedding = self.fusion([mix_vision_embedding, bert_embedding])
        predictions = self.classifier(final_embedding)

        return predictions, final_embedding, mix_vision_embedding, bert_embedding

    def get_variables(self):
        if not self.all_variables:  # is None, not initialized
            self.bert_variables = self.bert.trainable_variables
            self.num_bert = len(self.bert_variables)
            self.normal_variables = self.nextvlad_1.trainable_variables + self.fusion.trainable_variables + \
                                    self.nextvlad_2.trainable_variables + self.nextvlad_3.trainable_variables + \
                                    self.bn.trainable_variables + self.mix_weights.trainable_variables + \
                                    self.classifier.trainable_variables + self.bert_map.trainable_variables # 这个之前忘记加了
            self.all_variables = self.bert_variables + self.normal_variables
        return self.all_variables

    def optimize(self, gradients):
        bert_gradients = gradients[:self.num_bert]
        self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
        normal_gradients = gradients[self.num_bert:]
        self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))


# class MultiModal_mlm(Model):
#     def __init__(self, config, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.bert = TFBertModel.from_pretrained(config.bert_dir)
#         # mlm head
#         bert_config = self.bert.config
#         self.mlm = TFBertMLMHead(bert_config, input_embeddings=self.bert.bert.embeddings, name="mlm___cls")

#         self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
#                                  output_size=config.vlad_hidden_size, dropout=config.dropout)
#         self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)
#         self.num_labels = config.num_labels
#         self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')
#         self.bert_map = tf.keras.layers.Dense(1024, activation ='relu')

#         self.bert_optimizer, self.bert_lr = create_optimizer(init_lr=config.bert_lr,
#                                                              num_train_steps=config.bert_total_steps,
#                                                              num_warmup_steps=config.bert_warmup_steps)
#         self.optimizer, self.lr = create_optimizer(init_lr=config.lr,
#                                                    num_train_steps=config.total_steps,
#                                                    num_warmup_steps=config.warmup_steps)
#         self.bert_variables, self.num_bert, self.normal_variables, self.all_variables = None, None, None, None

#     def call(self, inputs, training, **kwargs):
#         bert_output = self.bert([inputs['input_ids'], inputs['mask']]) # inputs have random mask
#         sequence_output = bert_output[0]
#         bert_embedding = bert_output[1]
#         prediction_scores_mlm = self.mlm(sequence_output=sequence_output, training=training)
#         # loss_mlm = (
#         #     None if inputs["mask_labels"] is None else self.compute_loss(labels=inputs["mask_labels"], logits=prediction_scores)
#         # )
#         bert_embedding = self.bert_map(bert_embedding)
#         frame_num = tf.reshape(inputs['num_frames'], [-1])
#         vision_embedding = self.nextvlad([inputs['frames'], frame_num])
#         vision_embedding = vision_embedding * tf.cast(tf.expand_dims(frame_num, -1) > 0, tf.float32)
#         final_embedding = self.fusion([vision_embedding, bert_embedding])
#         predictions = self.classifier(final_embedding)

#         return predictions, final_embedding, vision_embedding, bert_embedding, prediction_scores_mlm

#     def compute_loss(self, labels, logits):
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
#             from_logits=True, reduction=tf.keras.losses.Reduction.NONE
#         )
#         # make sure only labels that are not equal to -100 affect the loss
#         active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
#         reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
#         labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
#         return loss_fn(labels, reduced_logits)

#     def get_variables(self):
#         if not self.all_variables:  # is None, not initialized
#             self.bert_variables = self.bert.trainable_variables
#             self.num_bert = len(self.bert_variables)
#             self.normal_variables = self.nextvlad.trainable_variables + self.fusion.trainable_variables + \
#                                     self.classifier.trainable_variables + self.bert_map.trainable_variables + \
#                                     self.mlm.trainable_variables
#             self.all_variables = self.bert_variables + self.normal_variables
#         return self.all_variables

#     def optimize(self, gradients):
#         bert_gradients = gradients[:self.num_bert]
#         self.bert_optimizer.apply_gradients(zip(bert_gradients, self.bert_variables))
#         normal_gradients = gradients[self.num_bert:]
#         self.optimizer.apply_gradients(zip(normal_gradients, self.normal_variables))