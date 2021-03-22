

CUDA_VISIBLE_DEVICES=1 python extract_features.py \
--input_file="./data/test_text" \
--output_file="./data/test.jsonl" \
--vocab_file="./chinese_L-12_H-768_A-12/vocab.txt" \
--bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json" \
--init_checkpoint="./chinese_L-12_H-768_A-12/bert_model.ckpt" \
--layers=-2 \
--max_seq_length=128 \
--batch_size=8