mkdir 10fold_b_json
mkdir 10fold_b_zip
# MixNextvlad
python inference_pair_b.py --ckpt-file save/10fold/10fold_1_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_1_mix.zip --output-json 10fold_b_json/10fold_1_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_2_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_2_mix.zip --output-json 10fold_b_json/10fold_2_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_3_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_3_mix.zip --output-json 10fold_b_json/10fold_3_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_4_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_4_mix.zip --output-json 10fold_b_json/10fold_4_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_5_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_5_mix.zip --output-json 10fold_b_json/10fold_5_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_6_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_6_mix.zip --output-json 10fold_b_json/10fold_6_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_7_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_7_mix.zip --output-json 10fold_b_json/10fold_7_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_8_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_8_mix.zip --output-json 10fold_b_json/10fold_8_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_9_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_9_mix.zip --output-json 10fold_b_json/10fold_9_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_10_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_10_mix.zip --output-json 10fold_b_json/10fold_10_mix.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_11_mix/ckpt-4012 --output-zip 10fold_b_zip/10fold_11_mix.zip --output-json 10fold_b_json/10fold_11_mix.json

# MixNextvlad_ASL
python inference_pair_b.py --ckpt-file save/10fold/10fold_1_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_1_mix_asl.zip --output-json 10fold_b_json/10fold_1_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_2_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_2_mix_asl.zip --output-json 10fold_b_json/10fold_2_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_3_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_3_mix_asl.zip --output-json 10fold_b_json/10fold_3_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_4_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_4_mix_asl.zip --output-json 10fold_b_json/10fold_4_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_5_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_5_mix_asl.zip --output-json 10fold_b_json/10fold_5_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_6_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_6_mix_asl.zip --output-json 10fold_b_json/10fold_6_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_7_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_7_mix_asl.zip --output-json 10fold_b_json/10fold_7_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_8_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_8_mix_asl.zip --output-json 10fold_b_json/10fold_8_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_9_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_9_mix_asl.zip --output-json 10fold_b_json/10fold_9_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_10_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_10_mix_asl.zip --output-json 10fold_b_json/10fold_10_mix_asl.json
python inference_pair_b.py --ckpt-file save/10fold/10fold_11_mix_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_11_mix_asl.zip --output-json 10fold_b_json/10fold_11_mix_asl.json

# MixNextvlad_roformer
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_1_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_1_mix_roformer.zip --output-json 10fold_b_json/10fold_1_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_2_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_2_mix_roformer.zip --output-json 10fold_b_json/10fold_2_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_3_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_3_mix_roformer.zip --output-json 10fold_b_json/10fold_3_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_4_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_4_mix_roformer.zip --output-json 10fold_b_json/10fold_4_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_5_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_5_mix_roformer.zip --output-json 10fold_b_json/10fold_5_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_6_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_6_mix_roformer.zip --output-json 10fold_b_json/10fold_6_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_7_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_7_mix_roformer.zip --output-json 10fold_b_json/10fold_7_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_8_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_8_mix_roformer.zip --output-json 10fold_b_json/10fold_8_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_9_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_9_mix_roformer.zip --output-json 10fold_b_json/10fold_9_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_10_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_10_mix_roformer.zip --output-json 10fold_b_json/10fold_10_mix_roformer.json
python inference_pair_roformer_b.py --bert-dir junnyu/roformer_chinese --bert-seq-length 32 --ckpt-file save/10fold/10fold_11_mix_roformer/ckpt-4012 --output-zip 10fold_b_zip/10fold_11_mix_roformer.zip --output-json 10fold_b_json/10fold_11_mix_roformer.json

# Uniter
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_1_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_1_uniter.zip --output-json 10fold_b_json/10fold_1_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_2_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_2_uniter.zip --output-json 10fold_b_json/10fold_2_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_3_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_3_uniter.zip --output-json 10fold_b_json/10fold_3_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_4_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_4_uniter.zip --output-json 10fold_b_json/10fold_4_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_5_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_5_uniter.zip --output-json 10fold_b_json/10fold_5_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_6_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_6_uniter.zip --output-json 10fold_b_json/10fold_6_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_7_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_7_uniter.zip --output-json 10fold_b_json/10fold_7_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_8_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_8_uniter.zip --output-json 10fold_b_json/10fold_8_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_9_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_9_uniter.zip --output-json 10fold_b_json/10fold_9_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_10_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_10_uniter.zip --output-json 10fold_b_json/10fold_10_uniter.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_11_uniter/ckpt-3014 --output-zip 10fold_b_zip/10fold_11_uniter.zip --output-json 10fold_b_json/10fold_11_uniter.json

# Uniter_ASL
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_1_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_1_uniter_asl.zip --output-json 10fold_b_json/10fold_1_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_2_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_2_uniter_asl.zip --output-json 10fold_b_json/10fold_2_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_3_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_3_uniter_asl.zip --output-json 10fold_b_json/10fold_3_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_4_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_4_uniter_asl.zip --output-json 10fold_b_json/10fold_4_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_5_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_5_uniter_asl.zip --output-json 10fold_b_json/10fold_5_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_6_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_6_uniter_asl.zip --output-json 10fold_b_json/10fold_6_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_7_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_7_uniter_asl.zip --output-json 10fold_b_json/10fold_7_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_8_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_8_uniter_asl.zip --output-json 10fold_b_json/10fold_8_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_9_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_9_uniter_asl.zip --output-json 10fold_b_json/10fold_9_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_10_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_10_uniter_asl.zip --output-json 10fold_b_json/10fold_10_uniter_asl.json
python inference_pair_uniter_b.py --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_11_uniter_asl/ckpt-3014 --output-zip 10fold_b_zip/10fold_11_uniter_asl.zip --output-json 10fold_b_json/10fold_11_uniter_asl.json

# Uniter_roformer
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_1_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_1_uniter_roformer.zip --output-json 10fold_b_json/10fold_1_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_2_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_2_uniter_roformer.zip --output-json 10fold_b_json/10fold_2_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_3_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_3_uniter_roformer.zip --output-json 10fold_b_json/10fold_3_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_4_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_4_uniter_roformer.zip --output-json 10fold_b_json/10fold_4_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_5_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_5_uniter_roformer.zip --output-json 10fold_b_json/10fold_5_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_6_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_6_uniter_roformer.zip --output-json 10fold_b_json/10fold_6_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_7_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_7_uniter_roformer.zip --output-json 10fold_b_json/10fold_7_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_8_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_8_uniter_roformer.zip --output-json 10fold_b_json/10fold_8_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_9_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_9_uniter_roformer.zip --output-json 10fold_b_json/10fold_9_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_10_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_10_uniter_roformer.zip --output-json 10fold_b_json/10fold_10_uniter_roformer.json
python inference_pair_uniter_roformer_b.py --bert-dir junnyu/roformer_chinese_base --uniter-pooling mean --bert-seq-length 32 --ckpt-file save/10fold/10fold_11_uniter_roformer/ckpt-4013 --output-zip 10fold_b_zip/10fold_11_uniter_roformer.zip --output-json 10fold_b_json/10fold_11_uniter_roformer.json