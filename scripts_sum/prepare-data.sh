#!/usr/bin/env bash
# bash prepare-data.sh exp_test cnndm
exp_path=$1
data=$2  # cnndm, wikihow, xsum
model_size=large  # base, large
bart_path=./bart.$model_size

# fusion index threshold for generating oracle style label
if [ $data == cnndm ]; then
  abs_threshold=0.7
elif [ $data == xsum ]; then
  abs_threshold=0.7
elif [ $data == wikihow ]; then
  abs_threshold=0.3
else
  echo Unknown dataset $data.
  exit
fi

echo `date`, exp_path: $exp_path, data: $data
tok_path=$exp_path/$data.tokenized

echo `date`, Prepraring tokenized data...
python scripts_sum/make_datafiles.py --dataset $data --sep ' <SEP> ' --res $tok_path

seg_path=$exp_path/$data.segmented
enc_path=$exp_path/$data.encoded
bin_path=$exp_path/$data.binarized

echo `date`, Prepraring segmented data for ${data} with abs${abs_threshold}...
for D in test valid train; do #
  python scripts_sum/data_builder.py --workers 12 \
         --encoder-json $bart_path/encoder.json --vocab-bpe $bart_path/vocab.bpe \
         --data-format switch --source $tok_path/$D.source --target $tok_path/$D.target \
         --abs-threshold $abs_threshold --outdir-tok $seg_path --outdir-id $enc_path
done

echo `date`, Prepraring binarized data for ${data} with abs${abs_threshold}...
mkdir -p $bin_path
cp $bart_path/dict.txt $bin_path/dict.txt -f
cat scripts_sum/dict_tags.txt >> $bin_path/dict.txt
python -m fairseq_cli.preprocess --source-lang source --target-lang target --workers 12 \
       --trainpref $enc_path/train --validpref $enc_path/valid  \
       --destdir $bin_path --srcdict $bin_path/dict.txt --tgtdict $bin_path/dict.txt

