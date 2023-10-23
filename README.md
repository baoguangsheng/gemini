# GEMINI
This code is for EMNLP 2023 long paper "GEMINI: Controlling The Sentence-Level Summary Style
in Abstractive Text Summarization".

[Paper](https://arxiv.org/abs/2304.03548) 

## Brief Intro
Human take different styles in writing each summary sentence, either extractive style or abstractive style.
In this paper, we aim to mimic human summary styles, which we believe can increase our ability to control the style and deepen our understanding of how human summaries are produced.
We propose an adaptive model, GEMINI, which contains a rewriter and a generator to imitate the extractive style and the abstractive style, respectively.


## Prepare Data
We experiment on cnndm, xsum, and wikihow.
```
bash scripts_sum/prepare-data.sh exp_root cnndm
```

## Training Model
We use four Tesla V100 GPUs to train our model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_sum/run-bart-large.sh exp_root cnndm
```

## Evaluate Model
```
CUDA_VISIBLE_DEVICES=0 bash scripts_sum/test-rewriter.sh cnndm exp_root
```
