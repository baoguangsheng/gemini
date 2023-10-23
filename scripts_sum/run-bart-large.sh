#!/usr/bin/env bash
# bash run-bart-large.sh exp_test wikihow
exp_path=$1
data=$2  # cnndm, wikihow
model_size=large  # base, large
freeze_pretrain=8
bart_path=./bart.$model_size

echo `date`, data: $data, exp_path: $exp_path
tok_path=$exp_path/$data.tokenized
bin_path=$exp_path/$data.binarized

run_path=$exp_path/run-default
cp_path=$run_path/$data.checkpoints

echo `date`, run path: $run_path
mkdir -p $run_path

echo `date`, Training model...
python train.py  $bin_path --save-dir $cp_path --tensorboard-logdir $cp_path --seed 666 --num-workers 8 --fp16 \
       --task translation_gemini --arch bart_${model_size} --source-lang source --target-lang target --truncate-source \
       --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --skip-invalid-size-inputs-valid-test \
       --criterion label_smoothed_cross_entropy_gemini --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.00 --clip-norm 0.1  \
       --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 1000 --total-num-update 80000 \
       --max-tokens 2048 --update-freq 4 --validate-interval 1 --patience 2 --report-accuracy \
       --find-unused-parameters --no-epoch-checkpoints --no-last-checkpoints --required-batch-size-multiple 1 \
       --restore-file $bart_path/model.pt --reset-optimizer --reset-dataloader --reset-meters \
       --freeze-pretrain 8 --lrscale-randinit 10.0 --data-format switch \
       > $run_path/train.$data.log 2>&1

# copy tokenizer files
cp $bart_path/encoder.json $cp_path/. -f
cp $bart_path/vocab.bpe $cp_path/. -f
cp $bin_path/dict.* $cp_path/. -f

