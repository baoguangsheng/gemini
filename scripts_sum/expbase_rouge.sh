#!/usr/bin/env bash
# bash expbase_rouge reference system
ref=$1
sys=$2

echo expbase_rouge.sh
echo `date`, Calculating ROUGE...
export CLASSPATH=../../tools/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

cat $sys | sed -e 's/ <SEP> /<q>/g' | \
java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $sys.tmp

cat $ref | sed -e 's/ <SEP> /<q>/g' | \
java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $ref.tmp

python scripts_sum/bert_extractors/src/exp_base.py -gold $ref.tmp -candi $sys.tmp
