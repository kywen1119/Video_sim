# echo fold1! 0-5999val
# python train_pair.py --savedmodel-path save/10fold_1 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# python train_pair.py --savedmodel-path save/10fold_2 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# python train_pair.py --savedmodel-path save/10fold_3 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# python train_pair.py --savedmodel-path save/10fold_4 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# python train_pair.py --savedmodel-path save/10fold_5 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# python train_pair.py --savedmodel-path save/10fold_6 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# python train_pair.py --savedmodel-path save/10fold_7 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# python train_pair.py --savedmodel-path save/10fold_8 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# python train_pair.py --savedmodel-path save/10fold_9 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# python train_pair.py --savedmodel-path save/10fold_10 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# python train_pair.py --savedmodel-path save/10fold_11 --pretrain_model_dir save/pair_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_1_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_2_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_3_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_4_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_5_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_6_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_7_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_8_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_9_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_10_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# python train_pair.py --batch-size 100 --savedmodel-path save/10fold_11_addtf --pretrain_model_dir save/pair_bn_add_tf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_1_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_2_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_3_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_4_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_5_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_6_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_7_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_8_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_9_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_10_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# python train_pair.py --batch-size 112 --bert-dir hfl/chinese-macbert-base --savedmodel-path save/10fold_11_mac --pretrain_model_dir save/pair_macbert_bn --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_1_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_2_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_3_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_4_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_5_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_6_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_7_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_8_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_9_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# python train_pair_mix.py --batch-size 94 --savedmodel-path save/10fold_10_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# python train_pair_mix.py --batch-size 94 --savedmodel-path save/10fold_11_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_1_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_2_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_3_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_4_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_5_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_6_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_7_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_8_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_9_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128 --savedmodel-path save/10fold_10_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 128 --savedmodel-path save/10fold_11_mix_roformer_256 --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# # echo fold1! 0-5999val
# # CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_1_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# # echo fold2! 6000-11999val
# # CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_2_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# # echo fold3! 12000-17999val
# # CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_3_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_4_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_5_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# # echo fold6! 30000-35999val
# # CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_6_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_7_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_8_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# # echo fold9! 48000-53999val
# # CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128  --savedmodel-path save/10fold_9_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# # echo fold10! 54000-59999val
# # CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128 --savedmodel-path save/10fold_10_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix_asl.py --batch-size 128 --savedmodel-path save/10fold_11_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_1_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_2_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_3_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_4_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_5_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_6_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_7_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_8_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_9_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128 --savedmodel-path save/10fold_10_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128 --savedmodel-path save/10fold_11_mix5_asl1_256 --pretrain_model_dir save/mix5_asl1 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_1_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_2_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_3_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_4_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_5_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_6_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_7_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_8_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128  --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_9_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_10_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --hidden-size 1024 --savedmodel-path save/10fold_11_mix_1024_60 --pretrain_model_dir save/mix_1024_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord


# CUDA_VISIBLE_DEVICES=2 nohup python train_pair_uniter_tag.py --batch-size 128 --pretrain_model_dir save/uniter_mlm_tag_256_mean --uniter-pooling mean  --savedmodel-path save/ft_uniter_mlm_tag_mean_kl_3000 --total-steps 3000 --kl-weight 0.5

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord


# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --savedmodel-path save/10fold/10fold_1_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_addtf.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_mix_addtf --pretrain_model_dir save/mix_addtf --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord



# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling max --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_max_kl --pretrain_model_dir save/uniter_mlm_tag_256_max --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling cls --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_cls_kl --pretrain_model_dir save/uniter_mlm_tag_256_cls --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold_1_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_roformer_huber.py --bert-dir junnyu/roformer_chinese_base --batch-size 128  --savedmodel-path save/10fold_1_mix_roformer_256_huber --pretrain_model_dir save/mix_roformer_2 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# CUDA_VISIBLE_DEVICES=0 python train_pair_mix.py --batch-size 128  --savedmodel-path save/10fold_1_mix_nextsoft_filter --pretrain_model_dir save/nextsoft_mix --vlad-cluster-size 4000 --vlad-groups 4 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/filter_10/train.tfrecord --val-record-pattern data/pairwise/filter_10/val.tfrecord

# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling cls --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_vlad_kl --pretrain_model_dir save/uniter_mlm_tag_256_vlad --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord


# echo fold1! 0-7val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_1_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/0-7/train.tfrecord --val-record-pattern data/pairwise/filter_15/0-7/val.tfrecord
# echo fold2! 7-14
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_2_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/7-14/train.tfrecord --val-record-pattern data/pairwise/filter_15/7-14/val.tfrecord
# echo fold3! 14-21
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_3_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/14-21/train.tfrecord --val-record-pattern data/pairwise/filter_15/14-21/val.tfrecord
# echo fold4! 21-28
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_4_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/21-28/train.tfrecord --val-record-pattern data/pairwise/filter_15/21-28/val.tfrecord
# echo fold5! 28-35
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_5_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/28-35/train.tfrecord --val-record-pattern data/pairwise/filter_15/28-35/val.tfrecord
# echo fold6! 35-42
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_6_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/35-42/train.tfrecord --val-record-pattern data/pairwise/filter_15/35-42/val.tfrecord
# echo fold7! 42-49
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_7_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/42-49/train.tfrecord --val-record-pattern data/pairwise/filter_15/42-49/val.tfrecord
# echo fold8! 49-56
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_8_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/49-56/train.tfrecord --val-record-pattern data/pairwise/filter_15/49-56/val.tfrecord
# echo fold9! 56-63
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_9_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/56-63/train.tfrecord --val-record-pattern data/pairwise/filter_15/56-63/val.tfrecord
# echo fold10! 63-70
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/filter15_10_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/filter_15/63-70/train.tfrecord --val-record-pattern data/pairwise/filter_15/63-70/val.tfrecord

# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold_1_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=0 python train_pair_uniter_tag_mlm.py --batch-size 120 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter_mlm_tag_mean_kl_mlm --pretrain_model_dir save/uniter_mlm_tag_256_mean --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_1_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_2_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_3_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_4_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_5_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_6_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_7_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999vals
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_8_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_9_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_10_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix.py --batch-size 128 --bert-seq-len 60 --savedmodel-path save/10fold_11_mix_60 --pretrain_model_dir save/mix_60 --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# CUDA_VISIBLE_DEVICES=1 nohup python cqrtrain_mlm_mm_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 200 --total-steps 50000 --savedmodel-path save/uniter_mlm_tag_256_mean_roformer --uniter-pooling mean --bert-lr 5e-5 --bert-seq-length 32 > run_uniter_mlm_tag_256_mean_roformer.log 2>&1 &

# echo fold1! 0-5999val
# yes CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# yes CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# yes CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# yes CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_uniter_tag_asl.py --batch-size 128 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter_mlm_tag_mean_kl --pretrain_model_dir save/uniter_mlm_tag_256_mean_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord


# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# CUDA_VISIBLE_DEVICES=1 python train_pair_uniter_tag_roformer.py --bert-dir junnyu/roformer_chinese_base --batch-size 112 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_uniter_mlm_tag_mean_kl_roformer --pretrain_model_dir save/uniter_mlm_tag_256_mean_roformer --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# CUDA_VISIBLE_DEVICES=2 nohup python cqrtrain_mix_roformer.py --batch-size 256 --bert-dir junnyu/roformer_chinese_base --total-steps 50000 --savedmodel-path save/mix_roformer_asl > run_mix_roformer_asl.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96  --savedmodel-path save/mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_1_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
# echo fold2! 6000-11999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_2_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
# echo fold3! 12000-17999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_3_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
# echo fold4! 18000-23999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_4_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
# echo fold5! 24000-29999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_5_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
# echo fold6! 30000-35999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_6_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
# echo fold7! 36000-41999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_7_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
# echo fold8! 42000-47999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_8_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
# echo fold9! 48000-53999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_9_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
# echo fold10! 54000-59999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_10_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
# echo fold11! 60000-65999val
# CUDA_VISIBLE_DEVICES=1 python train_pair_mix_rank.py --batch-size 96 --uniter-pooling mean --savedmodel-path save/10fold/10fold_11_mix_rank --pretrain_model_dir save/mix_3contra --kl-weight 0.1 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord

# echo fold1! 0-5999val
# CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_1_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_2_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_3_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_4_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_5_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_6_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_7_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_8_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
CUDA_VISIBLE_DEVICES=0 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_9_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_10_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
CUDA_VISIBLE_DEVICES=2 python train_pair_mix_roformer_asl.py --batch-size 128 --bert-dir junnyu/roformer_chinese_base --savedmodel-path save/10fold_11_mix_roformer_asl --pretrain_model_dir save/mix_roformer_asl --kl-weight 0.5 --total-steps 3000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord