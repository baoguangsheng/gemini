#!/usr/bin/env bash
# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# command help
if [ $# == '0' ]; then
    echo "Please follow the usage:"
    echo "    bash $0 cnndm exp_test"
    exit
fi

data=$1
exp_path=$2

echo `date`, exp_path: $exp_path, data: $data
tok_path=$exp_path/$data.tokenized
run_path=$exp_path/run-default
cp_path=$run_path/$data.checkpoints
res_path=$run_path/$data.results

echo `date`, Generate summary...
mkdir -p $res_path

if [ $data == cnndm ]; then
  beam_search="--batch-size 16 --beam 1 --max-len-a 0 --max-len-b 200 --min-len 20 --lenpen 1.0 --abspen 1.0"
elif [ $data == wikihow ]; then
  beam_search="--batch-size 8 --beam 3 --max-len-a 0 --max-len-b 200 --min-len 10 --lenpen 1.0 --abspen 3.0"
elif [ $data == xsum ]; then
  beam_search="--batch-size 4 --beam 5 --max-len-a 0 --max-len-b 100 --min-len 10 --lenpen 1.0 --abspen 1.0"
else
  echo Unknown dataset $data.
  exit
fi

split=test
rm $res_path/${split}.ref -f
rm $res_path/${split}.gen -f

python scripts_sum/rewriter.py $cp_path --data-format switch --source $tok_path/${split}.source --target $tok_path/${split}.target \
  --rewriter-path $cp_path --outdir $res_path  $beam_search \
  > $run_path/${split}.$data.log 2>&1

echo `date`, Calculating ROUGE...
cp $tok_path/${split}.target $res_path/${split}.ref -f
bash scripts_sum/expbase_rouge.sh $res_path/${split}.ref $res_path/${split}.gen \
  >> $run_path/${split}.$data.log 2>&1

