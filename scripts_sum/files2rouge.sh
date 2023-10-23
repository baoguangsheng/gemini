#!/usr/bin/env bash
# bash files2rouge reference system
ref=$1
sys=$2

echo files2rouge.sh
echo `date`, Calculating ROUGE...
export CLASSPATH=../../tools/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

cat $sys | sed -e 's/ <SEP> / /g' | \
java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $sys.tmp

cat $ref | sed -e 's/ <SEP> / /g' | \
java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > $ref.tmp

files2rouge $ref.tmp $sys.tmp
