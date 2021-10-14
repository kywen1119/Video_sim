### [Multimodal Video Similarity Challenge](https://algo.browser.qq.com/)
#### [@CIKM 2021](https://www.cikm2021.org/analyticup) 
#### Implementation source codes of team <618大庆神!>.
#### Final score: 82.8307 on test_b.

#### 1. 模型总览
我们的最终结果由6个模型的ensemble组成，先在开头概述这6个模型：
| Model   | test_a  | test_b | weight |
| MixNextvlad  | ----  |
| MixNextvlad_ASL  | 单元格 |
| MixNextvlad_roformer  | 单元格 |
| Uniter  | 单元格 |
| Uniter_asl  | 单元格 |
| Uniter_roformer  | 单元格 |
 <table>
        <tr>
            <th>Model</th>
            <th>test_a</th>
            <th>test_b</th>
            <th>weight</th>
        </tr>
        <tr>
            <th>MixNextvlad</th>
            <th>81.6</th>
            <th>x</th>
            <th>0.17</th>
        </tr>
        <tr>
            <th>MixNextvlad_ASL</th>
            <th>81.8</th>
            <th>x</th>
            <th>0.2</th>
        </tr>
        <tr>
            <th>MixNextvlad_roformer</th>
            <th>81.3</th>
            <th>x</th>
            <th>0.13</th>
        </tr>
        <tr>
            <th>Uniter</th>
            <th>81.4</th>
            <th>x</th>
            <th>0.13</th>
        </tr>
        <tr>
            <th>Uniter_asl</th>
            <th>81.6</th>
            <th>x</th>
            <th>0.2</th>
        </tr>
        <tr>
            <th>Uniter_roformer</th>
            <th>81.6</th>
            <th>x</th>
            <th>0.17</th>
        </tr>
    </table>

#### 2. 模型介绍
主要使用了两种模型，第一种是基于baseline改进的MixNextvald，在[1]中提出；第二种是基于transformer的Uniter [2]模型。 
##### 2.1 MixNextvlad
如图，论文中也有详细介绍，不再赘述。
##### 2.2 Uniter
如图，视频的帧feature和句子的单词embedding经过concat之后送入bert-encoder，输出的features取平均得到最后的embedding。使用该模型时预训练任务增加了MLM（masked language modeling），只对文本进行mask，然后通过上下文的文本和图像特征共同进行MLM。
##### 2.2 ASL
这是阿里巴巴最新提出的一种用于多标签分类的Loss [3]，可以有效解决多标签分类长尾样本的噪声问题。用它来替换baseline中的多标签分类的BCE损失，可以使得收敛更快，最终F1 score也更高。
##### 2.2 Roformer
用来替换bert。[4]

#### 3. 一些tricks
总体来说：先进性pretrain（多标签分类 or MLM），再进行finetune （MSE）.
##### 针对预训练
1. 替换原来的bert model，baseline中的是bert-uncased-chinese，更换成更好的chinese-roberta-wwm-ext；或者更换为roformer_chinese_base。
2. 对于MixNextvlad 模型，在文本特征和图像特征进行fusion前增加一个对比损失函数（contrastive loss），我们认为这样能平衡二者的量纲，能促进fusion的效果，在pretrain时能有效提升spearman 3个百分点。
3. 

##### 针对finetune
1. 使用11-fold-cross-validation：将pairwise的数据分成11份，每次用一份进行验证，这样同一个模型可以训11个模型，最终embedding取平均。
2. finetune时使用三个损失函数，包括：mse loss （直接优化similarity）、KL loss （优化模型得到的sim和label sim的分布差距）、tag loss （也就是预训练的多标签分类loss）。单独使用mse会过拟合，增加后两个之后可以有效缓解。
3. 训练轮次不宜过多，我们128的batch size只需要训练 3000-4000 steps。

#### 4. 如何复现？
所有实验在一块3090上完成。
环境： tensorflow==2.5.0    transformers    
##### 4.1 数据准备
生成pair对的tfrecord，且有11个文件（11-fold-cross-validation）
```bash
  python write_tfrecord.py
```
生成的结果：（每个路径下面包含 train.tfrecord & val.tfrecord）
```
├── data/
|   ├── pairwise/           
|   |   ├── 0-5999val/
|   |   ├── 6000-11999val/
|   |   ├── 12000-17999val/
|   |   ├── 18000-23999val/
|   |   ├── 24000-29999val/
|   |   ├── 30000-35999val/
|   |   ├── 36000-41999val/
|   |   ├── 42000-47999val/
|   |   ├── 48000-53999val/
|   |   ├── 54000-59999val/
|   |   ├── 60000-65999val/
```

##### 4.2 模型预训练
+ Pre-Train on MixNextvlad models:
    + MixNextvlad:
    ```bash
    python cqrtrain_mix.py --batch-size 256 --savedmodel-path save/mix
    ```
    + MixNextvlad_ASL:
    ```bash
    python cqrtrain_mix_asl.py --batch-size 256 --savedmodel-path save/mix_asl
    ```
    + MixNextvlad_roformer:
    ```bash
    python cqrtrain_mix_roformer.py --batch-size 256 --savedmodel-path save/mix_roformer --bert-dir junnyu/roformer_chinese_base
    ```
+ Pre-Train on Uniter models:
    + Uniter:
    ```bash
    python cqrtrain_mlm_mm_tag.py --batch-size 256 --savedmodel-path save/uniter --uniter-pooling mean 
    ```
    + Uniter_ASL:
    ```bash
    python cqrtrain_mlm_mm_tag_asl.py --batch-size 256 --savedmodel-path save/uniter_asl --uniter-pooling mean 
    ```
    + Uniter_roformer:
    ```bash
    python cqrtrain_mlm_mm_tag_roformer.py --batch-size 210 --savedmodel-path save/uniter_roformer --uniter-pooling mean --bert-dir junnyu/roformer_chinese_base 
    ```
##### 4.3 模型finetune
注意！在训练ASL的时候有可能会出现NAN，这种情况需要重跑一次相应的模型.
建议每次只跑sh文件里面的一个模型，把其他的注释掉，这样方便debug。
```bash
sh finetune_all.sh
```
##### 4.4 模型inference
建议每次只跑sh文件里面的一个模型，把其他的注释掉，这样方便debug。
```bash
sh infer_all.sh
```
##### 4.5 ensemble
```bash
python ensemble_final.py
```
#### 5. 一些无用的尝试

#### 6. References
[1] Lin R, Xiao J, Fan J. Nextvlad: An efficient neural network to aggregate frame-level features for large-scale video classification[C]//Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 2018: 0-0.
[2] Chen Y C, Li L, Yu L, et al. Uniter: Universal image-text representation learning[C]//European conference on computer vision. Springer, Cham, 2020: 104-120.
[3] Ridnik T, Ben-Baruch E, Zamir N, et al. Asymmetric Loss for Multi-Label Classification[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 82-91.
[4] Su J, Lu Y, Pan S, et al. Roformer: Enhanced transformer with rotary position embedding[J]. arXiv preprint arXiv:2104.09864, 2021.
