train model with custom segment embedding
```bash
nohup python title_compression/token_cls.py --task_name tc --do_train --do_eval --do_lower_case --data_dir ../../data/title_compression  --bert_model ../../data/title_compression/bies_pretrained  --max_seq_length 64  --train_batch_size 32 --eval_batch_size 32 --type_vocab_size 4 --learning_rate 2e-5 --eval_step 5000  --num_train_epochs 10.0   --output_dir tmp/tc_output/ 2>&1 > log.txt &
```