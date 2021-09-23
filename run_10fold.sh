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

echo fold1! 0-5999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_1_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/0-5999val/train.tfrecord --val-record-pattern data/pairwise/0-5999val/val.tfrecord
echo fold2! 6000-11999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_2_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/6000-11999val/train.tfrecord --val-record-pattern data/pairwise/6000-11999val/val.tfrecord
echo fold3! 12000-17999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_3_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/12000-17999val/train.tfrecord --val-record-pattern data/pairwise/12000-17999val/val.tfrecord
echo fold4! 18000-23999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_4_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/18000-23999val/train.tfrecord --val-record-pattern data/pairwise/18000-23999val/val.tfrecord
echo fold5! 24000-29999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_5_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/24000-29999val/train.tfrecord --val-record-pattern data/pairwise/24000-29999val/val.tfrecord
echo fold6! 30000-35999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_6_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/30000-35999val/train.tfrecord --val-record-pattern data/pairwise/30000-35999val/val.tfrecord
echo fold7! 36000-41999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_7_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/36000-41999val/train.tfrecord --val-record-pattern data/pairwise/36000-41999val/val.tfrecord
echo fold8! 42000-47999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_8_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/42000-47999val/train.tfrecord --val-record-pattern data/pairwise/42000-47999val/val.tfrecord
echo fold9! 48000-53999val
python train_pair_mix.py --batch-size 94  --savedmodel-path save/10fold_9_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/48000-53999val/train.tfrecord --val-record-pattern data/pairwise/48000-53999val/val.tfrecord
echo fold10! 54000-59999val
python train_pair_mix.py --batch-size 94 --savedmodel-path save/10fold_10_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/54000-59999val/train.tfrecord --val-record-pattern data/pairwise/54000-59999val/val.tfrecord
echo fold11! 60000-65999val
python train_pair_mix.py --batch-size 94 --savedmodel-path save/10fold_11_mix --pretrain_model_dir save/mix_3contra --kl-weight 0.5 --total-steps 4000 --train-record-pattern data/pairwise/60000-65999val/train.tfrecord --val-record-pattern data/pairwise/60000-65999val/val.tfrecord