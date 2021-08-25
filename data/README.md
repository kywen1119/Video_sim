#### description
- 'chinese_L-12_H-768_A-12' is the pretrained BERT model for [huggingface Transformers](https://github.com/huggingface/transformers). You can choose any open-source pretraining weights. 
- 'tag_list.txt' is a tag list demo for multi-label classification. You are encouraged to choose your own supervised signal including tags, categories and etc.

#### contents in 'data' folder in tree-like format
```
├── chinese_L-12_H-768_A-12  
│   ├── bert_config.json  
│   ├── bert_model.ckpt.data-00000-of-00001  
│   ├── bert_model.ckpt.index  
│   ├── bert_model.ckpt.meta  
│   ├── config.json  
│   ├── tf_model.h5  
│   └── vocab.txt  
├── pairwise  
│   ├── label.tsv  
│   └── pairwise.tfrecords  
├── pointwise  
│   ├── pretrain_0.tfrecords  
│   ├── pretrain_10.tfrecords  
│   ├── pretrain_11.tfrecords  
│   ├── pretrain_12.tfrecords  
│   ├── pretrain_13.tfrecords  
│   ├── pretrain_14.tfrecords  
│   ├── pretrain_15.tfrecords  
│   ├── pretrain_16.tfrecords  
│   ├── pretrain_17.tfrecords  
│   ├── pretrain_18.tfrecords  
│   ├── pretrain_19.tfrecords  
│   ├── pretrain_1.tfrecords  
│   ├── pretrain_2.tfrecords  
│   ├── pretrain_3.tfrecords  
│   ├── pretrain_4.tfrecords  
│   ├── pretrain_5.tfrecords  
│   ├── pretrain_6.tfrecords  
│   ├── pretrain_7.tfrecords  
│   ├── pretrain_8.tfrecords  
│   └── pretrain_9.tfrecords  
├── tag_list.txt  
└── test_a  
    └── test_a.tfrecords  
```