### 1. MixNextvlad
echo fold1! 0-5999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_1_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_2_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_3_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_4_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_5_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_6_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_7_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_8_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_9_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
python train_pair_mix.py --batch-size 128 --savedmodel-path save/10fold_10_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
python train_pair_mix.py --batch-size 128 --savedmodel-path save/10fold_11_mix --pretrain_model_dir save/mix --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

### 2. MixNextvlad_ASL
# 注意！在训练ASL的时候有可能会出现NAN，这种情况需要重跑一次相应的模型
echo fold1! 0-5999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_1_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_2_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_3_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_4_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_5_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_6_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_7_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_8_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_9_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128 --savedmodel-path save/10fold_10_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128 --savedmodel-path save/10fold_11_mix_asl --pretrain_model_dir save/mix_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

### 3. MixNextvlad_roformer
echo fold1! 0-5999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_1_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_2_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_3_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_4_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_5_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_6_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_7_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_8_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_9_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128 --savedmodel-path save/10fold_10_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128 --savedmodel-path save/10fold_11_mix_roformer --pretrain_model_dir save/mix_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

### 4. Uniter
echo fold1! 0-5999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter --pretrain_model_dir save/uniter --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

### 5. Uniter_ASL
# 注意！在训练ASL的时候有可能会出现NAN，这种情况需要重跑一次相应的模型
echo fold1! 0-5999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter_asl --pretrain_model_dir save/uniter_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

### 6. Uniter_roformer
echo fold1! 0-5999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter_roformer --pretrain_model_dir save/uniter_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord