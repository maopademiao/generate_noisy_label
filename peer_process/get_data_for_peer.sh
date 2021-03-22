# trec
python ./generate_noisy_label/peer_process/format_raw_data.py \
--input_file="./generate_noisy_label/data/trec/train.json" \
--output_label="./generate_noisy_label/data/trec/train_peer_label.txt" \
--output_sent="./generate_noisy_label/data/trec/train_peer_sent.txt"
CUDA_VISIBLE_DEVICES=0 python extract_features.py \
--input_file="./generate_noisy_label/data/trec/train_peer_sent.txt" \
--output_file="./generate_noisy_label/data/trec/train_peer_sent.jsonl" \
--vocab_file="./uncased_L-12_H-768_A-12/vocab.txt" \
--bert_config_file="./uncased_L-12_H-768_A-12/bert_config.json" \
--init_checkpoint="./uncased_L-12_H-768_A-12/bert_model.ckpt" \
--layers=-2 \
--max_seq_length=128 \
--batch_size=8
python ./generate_noisy_label/peer_process/make_data.py \
--input_label="./generate_noisy_label/data/trec/train_peer_label.txt" \
--input_emb="./generate_noisy_label/data/trec/train_peer_sent.jsonl" \
--output_file="./generate_noisy_label/data/trec/train_peer.json"

# agnews
python ./generate_noisy_label/peer_process/format_raw_data.py \
--input_file="./generate_noisy_label/data/agnews/train.json" \
--output_label="./generate_noisy_label/data/agnews/train_peer_label.txt" \
--output_sent="./generate_noisy_label/data/agnews/train_peer_sent.txt"
CUDA_VISIBLE_DEVICES=1 python extract_features.py \
--input_file="./generate_noisy_label/data/agnews/train_peer_sent.txt" \
--output_file="./generate_noisy_label/data/agnews/train_peer_sent.jsonl" \
--vocab_file="./uncased_L-12_H-768_A-12/vocab.txt" \
--bert_config_file="./uncased_L-12_H-768_A-12/bert_config.json" \
--init_checkpoint="./uncased_L-12_H-768_A-12/bert_model.ckpt" \
--layers=-2 \
--max_seq_length=128 \
--batch_size=8
python ./generate_noisy_label/peer_process/make_data.py \
--input_label="./generate_noisy_label/data/agnews/train_peer_label.txt" \
--input_emb="./generate_noisy_label/data/agnews/train_peer_sent.jsonl" \
--output_file="./generate_noisy_label/data/agnews/train_peer.json"

# chn
python ./generate_noisy_label/peer_process/format_raw_data.py \
--input_file="./generate_noisy_label/data/chn/train.json" \
--output_label="./generate_noisy_label/data/chn/train_peer_label.txt" \
--output_sent="./generate_noisy_label/data/chn/train_peer_sent.txt"
CUDA_VISIBLE_DEVICES=2 python extract_features.py \
--input_file="./generate_noisy_label/data/chn/train_peer_sent.txt" \
--output_file="./generate_noisy_label/data/chn/train_peer_sent.jsonl" \
--vocab_file="./chinese_L-12_H-768_A-12/vocab.txt" \
--bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json" \
--init_checkpoint="./chinese_L-12_H-768_A-12/bert_model.ckpt" \
--layers=-2 \
--max_seq_length=128 \
--batch_size=8
python ./generate_noisy_label/peer_process/make_data.py \
--input_label="./generate_noisy_label/data/chn/train_peer_label.txt" \
--input_emb="./generate_noisy_label/data/chn/train_peer_sent.jsonl" \
--output_file="./generate_noisy_label/data/chn/train_peer.json"

# chngolden
python ./generate_noisy_label/peer_process/format_raw_data.py \
--input_file="./generate_noisy_label/data/chngolden/train.json" \
--output_label="./generate_noisy_label/data/chngolden/train_peer_label.txt" \
--output_sent="./generate_noisy_label/data/chngolden/train_peer_sent.txt"
CUDA_VISIBLE_DEVICES=3 python extract_features.py \
--input_file="./generate_noisy_label/data/chngolden/train_peer_sent.txt" \
--output_file="./generate_noisy_label/data/chngolden/train_peer_sent.jsonl" \
--vocab_file="./chinese_L-12_H-768_A-12/vocab.txt" \
--bert_config_file="./chinese_L-12_H-768_A-12/bert_config.json" \
--init_checkpoint="./chinese_L-12_H-768_A-12/bert_model.ckpt" \
--layers=-2 \
--max_seq_length=128 \
--batch_size=8
python ./generate_noisy_label/peer_process/make_data.py \
--input_label="./generate_noisy_label/data/chngolden/train_peer_label.txt" \
--input_emb="./generate_noisy_label/data/chngolden/train_peer_sent.jsonl" \
--output_file="./generate_noisy_label/data/chngolden/train_peer.json"
