# """
# 对所有ckpt进行预测，直接得到所有模型的json，再进行ensemble，总共包含：
# 6个模型，每个模型是11-fold的训练，其中MixNextvlad_ASL只有8个模型，总共63个模型
# """

mkdir 10fold_b_json
mkdir 10fold_b_zip
# MixNextvlad
echo inference MixNextvlad!
python inference_pair_b.py --ckpt-file final_save/10fold_1_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_1_mix.zip --output-json 10fold_b_json/10fold_1_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_2_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_2_mix.zip --output-json 10fold_b_json/10fold_2_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_3_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_3_mix.zip --output-json 10fold_b_json/10fold_3_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_4_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_4_mix.zip --output-json 10fold_b_json/10fold_4_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_5_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_5_mix.zip --output-json 10fold_b_json/10fold_5_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_6_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_6_mix.zip --output-json 10fold_b_json/10fold_6_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_7_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_7_mix.zip --output-json 10fold_b_json/10fold_7_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_8_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_8_mix.zip --output-json 10fold_b_json/10fold_8_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_9_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_9_mix.zip --output-json 10fold_b_json/10fold_9_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_10_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_10_mix.zip --output-json 10fold_b_json/10fold_10_mix.json
python inference_pair_b.py --ckpt-file final_save/10fold_11_mix/ckpt-4014 --output-zip 10fold_b_zip/10fold_11_mix.zip --output-json 10fold_b_json/10fold_11_mix.json

# MixNextvlad_ASL
echo inference MixNextvlad_ASL!
python inference_pair_b.py --ckpt-file final_save/10fold_1_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_1_mix_asl.zip --output-json 10fold_b_json/10fold_1_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_2_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_2_mix_asl.zip --output-json 10fold_b_json/10fold_2_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_3_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_3_mix_asl.zip --output-json 10fold_b_json/10fold_3_mix_asl.json
# python inference_pair_b.py --ckpt-file final_save/10fold_4_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_4_mix_asl.zip --output-json 10fold_b_json/10fold_4_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_5_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_5_mix_asl.zip --output-json 10fold_b_json/10fold_5_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_6_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_6_mix_asl.zip --output-json 10fold_b_json/10fold_6_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_7_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_7_mix_asl.zip --output-json 10fold_b_json/10fold_7_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_8_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_8_mix_asl.zip --output-json 10fold_b_json/10fold_8_mix_asl.json
# python inference_pair_b.py --ckpt-file final_save/10fold_9_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_9_mix_asl.zip --output-json 10fold_b_json/10fold_9_mix_asl.json
python inference_pair_b.py --ckpt-file final_save/10fold_10_mix_asl/ckpt-3016 --output-zip 10fold_b_zip/10fold_10_mix_asl.zip --output-json 10fold_b_json/10fold_10_mix_asl.json
# python inference_pair_b.py --ckpt-file final_save/10fold_11_mix_asl/ckpt-2517 --output-zip 10fold_b_zip/10fold_11_mix_asl.zip --output-json 10fold_b_json/10fold_11_mix_asl.json

# MixNextvlad_roformer
echo inference MixNextvlad_roformer!
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_1_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_1_mix_roformer.zip --output-json 10fold_b_json/10fold_1_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_2_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_2_mix_roformer.zip --output-json 10fold_b_json/10fold_2_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_3_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_3_mix_roformer.zip --output-json 10fold_b_json/10fold_3_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_4_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_4_mix_roformer.zip --output-json 10fold_b_json/10fold_4_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_5_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_5_mix_roformer.zip --output-json 10fold_b_json/10fold_5_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_6_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_6_mix_roformer.zip --output-json 10fold_b_json/10fold_6_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_7_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_7_mix_roformer.zip --output-json 10fold_b_json/10fold_7_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_8_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_8_mix_roformer.zip --output-json 10fold_b_json/10fold_8_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_9_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_9_mix_roformer.zip --output-json 10fold_b_json/10fold_9_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_10_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_10_mix_roformer.zip --output-json 10fold_b_json/10fold_10_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese_base --bert-seq-length 32 --ckpt-file final_save/10fold_11_mix_roformer/ckpt-4015 --output-zip 10fold_b_zip/10fold_11_mix_roformer.zip --output-json 10fold_b_json/10fold_11_mix_roformer.json

# Uniter
echo inference Uniter!
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_1_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_1_uniter.zip --output-json 10fold_b_json/10fold_1_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_2_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_2_uniter.zip --output-json 10fold_b_json/10fold_2_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_3_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_3_uniter.zip --output-json 10fold_b_json/10fold_3_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_4_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_4_uniter.zip --output-json 10fold_b_json/10fold_4_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_5_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_5_uniter.zip --output-json 10fold_b_json/10fold_5_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_6_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_6_uniter.zip --output-json 10fold_b_json/10fold_6_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_7_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_7_uniter.zip --output-json 10fold_b_json/10fold_7_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_8_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_8_uniter.zip --output-json 10fold_b_json/10fold_8_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_9_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_9_uniter.zip --output-json 10fold_b_json/10fold_9_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_10_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_10_uniter.zip --output-json 10fold_b_json/10fold_10_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_11_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_11_uniter.zip --output-json 10fold_b_json/10fold_11_uniter.json

# Uniter_ASL
echo inference Uniter_ASL!
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_1_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_1_uniter_asl.zip --output-json 10fold_b_json/10fold_1_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_2_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_2_uniter_asl.zip --output-json 10fold_b_json/10fold_2_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_3_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_3_uniter_asl.zip --output-json 10fold_b_json/10fold_3_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_4_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_4_uniter_asl.zip --output-json 10fold_b_json/10fold_4_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_5_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_5_uniter_asl.zip --output-json 10fold_b_json/10fold_5_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_6_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_6_uniter_asl.zip --output-json 10fold_b_json/10fold_6_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_7_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_7_uniter_asl.zip --output-json 10fold_b_json/10fold_7_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_8_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_8_uniter_asl.zip --output-json 10fold_b_json/10fold_8_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_9_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_9_uniter_asl.zip --output-json 10fold_b_json/10fold_9_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_10_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_10_uniter_asl.zip --output-json 10fold_b_json/10fold_10_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_11_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_11_uniter_asl.zip --output-json 10fold_b_json/10fold_11_uniter_asl.json

# Uniter_roformer
echo inference Uniter_roformer!
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_1_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_1_uniter_roformer.zip --output-json 10fold_b_json/10fold_1_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_2_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_2_uniter_roformer.zip --output-json 10fold_b_json/10fold_2_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_3_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_3_uniter_roformer.zip --output-json 10fold_b_json/10fold_3_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_4_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_4_uniter_roformer.zip --output-json 10fold_b_json/10fold_4_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_5_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_5_uniter_roformer.zip --output-json 10fold_b_json/10fold_5_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_6_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_6_uniter_roformer.zip --output-json 10fold_b_json/10fold_6_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_7_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_7_uniter_roformer.zip --output-json 10fold_b_json/10fold_7_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_8_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_8_uniter_roformer.zip --output-json 10fold_b_json/10fold_8_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_9_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_9_uniter_roformer.zip --output-json 10fold_b_json/10fold_9_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_10_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_10_uniter_roformer.zip --output-json 10fold_b_json/10fold_10_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file final_save/10fold_11_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_11_uniter_roformer.zip --output-json 10fold_b_json/10fold_11_uniter_roformer.json

### 对所有embedding进行ensemble
echo ensemble!
python ensemble_final.py

# 最后输出为 result_10_b.zip