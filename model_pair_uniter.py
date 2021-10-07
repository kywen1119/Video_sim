import tensorflow as tf
from tensorflow.python.keras.models import Model
from transformers import TFBertModel, create_optimizer
from cqrmodel import SoftDBoF, NextSoftDBoF
from transformers.models.bert.modeling_tf_bert import TFBertMLMHead
from bert import TFBertModel_MM, shape_list

class MultiModal_Uniter(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel_MM.from_pretrained(config.bert_dir)
        # tf.print(self.bert)
        # mlm head
        bert_config = self.bert.config
        # self.mlm = TFBertMLMHead(bert_config, input_embeddings=self.bert.bert.embeddings, name="mlm___cls")

        # self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
        #                          output_size=config.vlad_hidden_size, dropout=config.dropout)
        # self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')
        self.frame_map = tf.keras.layers.Dense(768, activation ='relu')
        self.fc = tf.keras.layers.Dense(config.hidden_size)
        self.pooling = config.uniter_pooling

        self.bert_optimizer_1, self.bert_lr_1 = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer_1, self.lr_1 = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables_1, self.num_bert_1, self.normal_variables_1, self.all_variables_1 = None, None, None, None

    def call(self, inputs, training, **kwargs):
        image_embedding_1 = inputs['frames_1']
        _, num_segments_1, _ = image_embedding_1.shape
        image_embedding_1 = self.frame_map(image_embedding_1) # b,32,768
        frame_num_1 = tf.reshape(inputs['num_frames_1'], [-1])
        images_mask_1 = tf.sequence_mask(frame_num_1, maxlen=num_segments_1)
        images_mask_1 = tf.cast(images_mask_1, tf.int32)
        # _, seq_len = inputs['input_ids'].shape
        bert_output_1 = self.bert(input_ids=inputs['input_ids_1'], attention_mask=inputs['mask_1'], frame_features=image_embedding_1, frame_attention_mask=images_mask_1) # inputs have random mask
        sequence_output_1 = bert_output_1[0]
        sequence_output_1 = self.fc(sequence_output_1)
        if self.pooling == 'cls':
            bert_embedding_1 = sequence_output_1[:,0]
        elif self.pooling == 'mean':
            bert_embedding_1 = tf.reduce_mean(sequence_output_1, 1)
        elif self.pooling == 'max':
            text_mask_1 = 1-tf.cast(inputs['mask_1'], tf.int32)
            neg_mask_1 = tf.concat([text_mask_1, 1-images_mask_1], 1)
            super_neg_1 = tf.expand_dims(tf.cast(neg_mask_1, tf.float32), axis=2) * -1000
            bert_embedding_1 = tf.reduce_max(sequence_output_1 + super_neg_1, 1)

        # prediction_scores_mlm = self.mlm(sequence_output=sequence_output, training=training)[:,:seq_len]
        predictions_1 = self.classifier(bert_embedding_1)

        # 2
        image_embedding_2 = inputs['frames_2']
        _, num_segments_2, _ = image_embedding_2.shape
        image_embedding_2 = self.frame_map(image_embedding_2) # b,32,768
        frame_num_2 = tf.reshape(inputs['num_frames_2'], [-1])
        images_mask_2 = tf.sequence_mask(frame_num_2, maxlen=num_segments_2)
        images_mask_2 = tf.cast(images_mask_2, tf.int32)
        # _, seq_len = inputs['input_ids'].shape
        bert_output_2 = self.bert(input_ids=inputs['input_ids_2'], attention_mask=inputs['mask_2'], frame_features=image_embedding_2, frame_attention_mask=images_mask_2) # inputs have random mask
        sequence_output_2 = bert_output_2[0]
        sequence_output_2 = self.fc(sequence_output_2)
        if self.pooling == 'cls':
            bert_embedding_2 = sequence_output_2[:,0]
        elif self.pooling == 'mean':
            bert_embedding_2 = tf.reduce_mean(sequence_output_2, 1)
        elif self.pooling == 'max':
            text_mask_2 = 1-tf.cast(inputs['mask_2'], tf.int32)
            neg_mask_2 = tf.concat([text_mask_2, 1-images_mask_2], 1)
            super_neg_2 = tf.expand_dims(tf.cast(neg_mask_2, tf.float32), axis=2) * -1000
            bert_embedding_2 = tf.reduce_max(sequence_output_2 + super_neg_2, 1)
        # prediction_scores_mlm = self.mlm(sequence_output=sequence_output, training=training)[:,:seq_len]
        predictions_2 = self.classifier(bert_embedding_2)

        return bert_embedding_1, bert_embedding_2, predictions_1, predictions_2

    def get_variables(self):
        if not self.all_variables_1:  # is None, not initialized
            self.bert_variables_1 = self.bert.trainable_variables
            self.num_bert_1 = len(self.bert_variables_1)
            self.normal_variables_1 = self.frame_map.trainable_variables + \
                                    self.classifier.trainable_variables + self.fc.trainable_variables
            self.all_variables_1 = self.bert_variables_1 + self.normal_variables_1
        return self.all_variables_1

    def optimize(self, gradients):
        bert_gradients_1 = gradients[:self.num_bert_1]
        self.bert_optimizer_1.apply_gradients(zip(bert_gradients_1, self.bert_variables_1))
        normal_gradients_1 = gradients[self.num_bert_1:]
        self.optimizer_1.apply_gradients(zip(normal_gradients_1, self.normal_variables_1))


class MultiModal_Uniter_mlm(Model):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bert = TFBertModel_MM.from_pretrained(config.bert_dir)
        # tf.print(self.bert)
        # mlm head
        bert_config = self.bert.config
        self.mlm = TFBertMLMHead(bert_config, input_embeddings=self.bert.bert.embeddings, name="mlm___cls")

        # self.nextvlad = NeXtVLAD(config.frame_embedding_size, config.vlad_cluster_size,
        #                          output_size=config.vlad_hidden_size, dropout=config.dropout)
        # self.fusion = ConcatDenseSE(config.hidden_size, config.se_ratio)
        self.num_labels = config.num_labels
        self.classifier = tf.keras.layers.Dense(self.num_labels, activation='sigmoid')
        self.frame_map = tf.keras.layers.Dense(768, activation ='relu')
        self.fc = tf.keras.layers.Dense(config.hidden_size)
        self.pooling = config.uniter_pooling

        self.bert_optimizer_1, self.bert_lr_1 = create_optimizer(init_lr=config.bert_lr,
                                                             num_train_steps=config.bert_total_steps,
                                                             num_warmup_steps=config.bert_warmup_steps)
        self.optimizer_1, self.lr_1 = create_optimizer(init_lr=config.lr,
                                                   num_train_steps=config.total_steps,
                                                   num_warmup_steps=config.warmup_steps)
        self.bert_variables_1, self.num_bert_1, self.normal_variables_1, self.all_variables_1 = None, None, None, None

    def call(self, inputs, training, **kwargs):
        image_embedding_1 = inputs['frames_1']
        _, num_segments_1, _ = image_embedding_1.shape
        image_embedding_1 = self.frame_map(image_embedding_1) # b,32,768
        frame_num_1 = tf.reshape(inputs['num_frames_1'], [-1])
        images_mask_1 = tf.sequence_mask(frame_num_1, maxlen=num_segments_1)
        images_mask_1 = tf.cast(images_mask_1, tf.int32)
        _, seq_len_1 = inputs['input_ids_1'].shape
        bert_output_1 = self.bert(input_ids=inputs['input_ids_1'], attention_mask=inputs['mask_1'], frame_features=image_embedding_1, frame_attention_mask=images_mask_1) # inputs have random mask
        sequence_output_1 = bert_output_1[0]
        sequence_output_1 = self.fc(sequence_output_1)
        if self.pooling == 'cls':
            bert_embedding_1 = sequence_output_1[:,0]
        elif self.pooling == 'mean':
            bert_embedding_1 = tf.reduce_mean(sequence_output_1, 1)
        elif self.pooling == 'max':
            text_mask_1 = 1-tf.cast(inputs['mask_1'], tf.int32)
            neg_mask_1 = tf.concat([text_mask_1, 1-images_mask_1], 1)
            super_neg_1 = tf.expand_dims(tf.cast(neg_mask_1, tf.float32), axis=2) * -1000
            bert_embedding_1 = tf.reduce_max(sequence_output_1 + super_neg_1, 1)

        prediction_scores_mlm_1 = self.mlm(sequence_output=sequence_output_1, training=training)[:,:seq_len_1]
        predictions_1 = self.classifier(bert_embedding_1)

        # 2
        image_embedding_2 = inputs['frames_2']
        _, num_segments_2, _ = image_embedding_2.shape
        image_embedding_2 = self.frame_map(image_embedding_2) # b,32,768
        frame_num_2 = tf.reshape(inputs['num_frames_2'], [-1])
        images_mask_2 = tf.sequence_mask(frame_num_2, maxlen=num_segments_2)
        images_mask_2 = tf.cast(images_mask_2, tf.int32)
        _, seq_len_2 = inputs['input_ids_2'].shape
        bert_output_2 = self.bert(input_ids=inputs['input_ids_2'], attention_mask=inputs['mask_2'], frame_features=image_embedding_2, frame_attention_mask=images_mask_2) # inputs have random mask
        sequence_output_2 = bert_output_2[0]
        sequence_output_2 = self.fc(sequence_output_2)
        if self.pooling == 'cls':
            bert_embedding_2 = sequence_output_2[:,0]
        elif self.pooling == 'mean':
            bert_embedding_2 = tf.reduce_mean(sequence_output_2, 1)
        elif self.pooling == 'max':
            text_mask_2 = 1-tf.cast(inputs['mask_2'], tf.int32)
            neg_mask_2 = tf.concat([text_mask_2, 1-images_mask_2], 1)
            super_neg_2 = tf.expand_dims(tf.cast(neg_mask_2, tf.float32), axis=2) * -1000
            bert_embedding_2 = tf.reduce_max(sequence_output_2 + super_neg_2, 1)
        prediction_scores_mlm_2 = self.mlm(sequence_output=sequence_output_2, training=training)[:,:seq_len_2]
        predictions_2 = self.classifier(bert_embedding_2)

        return bert_embedding_1, bert_embedding_2, predictions_1, predictions_2, prediction_scores_mlm_1, prediction_scores_mlm_2

    def get_variables(self):
        if not self.all_variables_1:  # is None, not initialized
            self.bert_variables_1 = self.bert.trainable_variables
            self.num_bert_1 = len(self.bert_variables_1)
            self.normal_variables_1 = self.frame_map.trainable_variables + \
                                    self.classifier.trainable_variables + self.fc.trainable_variables
            self.all_variables_1 = self.bert_variables_1 + self.normal_variables_1 + self.mlm.trainable_variables
        return self.all_variables_1

    def optimize(self, gradients):
        bert_gradients_1 = gradients[:self.num_bert_1]
        self.bert_optimizer_1.apply_gradients(zip(bert_gradients_1, self.bert_variables_1))
        normal_gradients_1 = gradients[self.num_bert_1:]
        self.optimizer_1.apply_gradients(zip(normal_gradients_1, self.normal_variables_1))
