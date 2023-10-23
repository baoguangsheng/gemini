#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import json
import sys
import os

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.gpt2_bpe import get_encoder

from rouge import Rouge
import numpy as np
import scipy.stats as ss
import os.path as path
from tqdm import tqdm


def ext_index(recalls):
    return np.max(recalls)

def fusion_index(recalls, slen):
    def _smooth_recall(recalls, min_num, slen):
        recalls = list(recalls)
        if len(recalls) < min_num:
            recalls.extend([0] * (min_num - len(recalls)))
        return (np.array(recalls) * slen + 1) / (slen + 1)

    topk = 5
    recalls = _smooth_recall(recalls, topk, slen)
    recalls_topk = -np.partition(-recalls, topk - 1)[:topk]
    recalls_topk = recalls_topk / recalls_topk.sum()
    ent_recalls = ss.entropy(recalls_topk, base=2)
    ent_max = np.log2(topk)
    return ent_recalls / ent_max

def abs_index(recalls, slen):
    val = (1 - ext_index(recalls)) * fusion_index(recalls, slen)
    return val


''' Algorithm for matching closest sentence in article for each summary sentence
'''
def match_by_rouge12L(article, abstract, recall_threshold=0):
    def _match_recall(score):
        r = (score["rouge-1"]["r"] + score["rouge-2"]["r"] + score["rouge-l"]["r"]) / 3
        return r

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    match_recalls = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        scores = rouge.get_scores(hyps, refs)
        recalls = [_match_recall(score) for score in scores]
        match_recalls.append(recalls)
    # filter lower recall match
    match_recalls = np.array(match_recalls) * (np.array(match_recalls) > recall_threshold)
    return match_recalls


def match_by_rouge12L_custom(article, abstract, threshold):
    def _match_recall(score):
        r = (score["rouge-1"]["r"] + score["rouge-2"]["r"] + score["rouge-l"]["r"]) / 3
        return r

    def _match_score(score):
        r = (score["rouge-1"]["r"] + score["rouge-2"]["r"] + score["rouge-l"]["r"]) / 3
        p = (score["rouge-1"]["p"] + score["rouge-2"]["p"] + score["rouge-l"]["p"]) / 3
        return r + p

    def _rank_overall(scores, alpha=0):
        overall = np.mean(scores, axis=0)
        return np.array(scores) + alpha * overall

    rouge = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"])
    match_scores = []
    match_abss = []
    for sent in abstract:
        hyps = article
        refs = [sent] * len(article)
        try:
            scores = rouge.get_scores(hyps, refs)
            mrecalls = [_match_recall(score) for score in scores]
            mscores = [_match_score(score) for score in scores]
        except Exception as ex:
            print(ex)
            print('hyps:', hyps)
            print('refs:', refs)
            mrecalls = [0 for _ in range(len(refs))]
            mscores = [0 for _ in range(len(refs))]
        match_scores.append(mscores)
        match_abss.append([abs_index(mrecalls, len(sent.split()))])
    # add weight to recall of overall summary content
    match_scores = _rank_overall(match_scores, 0.4)
    # filter lower recall match
    match_scores = np.array(match_scores) * (np.array(match_abss) < threshold)
    return match_scores

class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.encoder_json, self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        line = ' ' + line.strip()
        ids = bpe.encode(line)
        return list(map(str, ids))

    def encode_lines(self, lines):
        assert len(lines) == 2, 'Lines: %s' % lines
        source, target = [line.strip() for line in lines]
        if len(source) == 0 or len(target) == 0:
            return ["EMPTY", None]

        sep = ' <SEP> '
        bos = '<S%s>'
        eos = '</S>'

        # match summary to source
        slines = source.split(sep)
        tlines = target.split(sep)

        # encode source and target
        sids = []
        stokens = []
        tids = []
        ttokens = []

        if self.args.data_format == 'previous': # the format required by previous BERT ContextRewriter
            abs_art_scores = match_by_rouge12L_custom(slines[:self.args.max_sent_index], tlines)
            abs_art_idx = np.argmax(abs_art_scores, axis=1).tolist()
            # source
            for idx, line in enumerate(slines):
                prefix = [bos % (tidx + 1) for tidx, sidx in enumerate(abs_art_idx) if sidx == idx]
                suffix = [eos] * len(prefix)
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
            # target
            for idx, line in enumerate(tlines):
                prefix = [bos % (idx + 1)]
                tids.extend(prefix + self.encode(line))
                ttokens.extend(prefix + line.split())
        elif self.args.data_format == 'reorder': # the format required by order sensitive rewriter
            abs_art_scores = match_by_rouge12L_custom(slines[:self.args.max_sent_index], tlines)
            abs_art_idx = np.argmax(abs_art_scores, axis=1).tolist()
            abs_reorder = [(tidx, sidx) for tidx, sidx in enumerate(abs_art_idx)]
            abs_reorder = list(sorted(abs_reorder, key=lambda x: x[1]))
            abs_reorder = [(tidx, sidx, gidx) for gidx, (tidx, sidx) in enumerate(abs_reorder)]
            abs_reorder = list(sorted(abs_reorder, key=lambda x: x[0]))
            # source
            for idx, line in enumerate(slines):
                prefix = [bos % (gidx + 1) for tidx, sidx, gidx in abs_reorder if sidx == idx]
                suffix = [eos] * len(prefix)
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
            # target
            for idx, line in enumerate(tlines):
                tidx, sidx, gidx = abs_reorder[idx]
                assert tidx == idx
                prefix = [bos % (gidx + 1)]
                tids.extend(prefix + self.encode(line))
                ttokens.extend(prefix + line.split())
        elif self.args.data_format == 'switch':  # the format required by rewriter-generator
            abs_art_scores = match_by_rouge12L_custom(slines[:self.args.max_sent_index], tlines, self.args.abs_threshold)
            abs_art_idx = np.argmax(abs_art_scores, axis=1)
            # source
            tok_gen = bos % ''
            sids.append(tok_gen)
            stokens.append(tok_gen)
            for idx, line in enumerate(slines):
                prefix = [bos % (idx + 1)] if idx < self.args.max_sent_index else []
                suffix = [eos] if idx < self.args.max_sent_index else []
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
            # target
            for idx, line in enumerate(tlines):
                sidx = abs_art_idx[idx]
                if abs_art_scores[idx, sidx] == 0.0:
                    prefix = [tok_gen]  # generate a new sentence
                else:
                    prefix = [bos % (sidx + 1)]  # rewrite the sentence
                suffix = [eos]
                tids.extend(prefix + self.encode(line) + suffix)
                ttokens.extend(prefix + line.split() + suffix)
        else:
            raise Exception()

        # output
        enc_lines = [sids, tids, stokens, ttokens]
        enc_lines = [' '.join(line) for line in enc_lines]
        return ["PASS", enc_lines]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        default='../shared/bart.base/encoder.json',
        help='path to encoder.json',
    )
    parser.add_argument(
        "--vocab-bpe",
        default='../shared/bart.base/vocab.bpe',
        help='path to vocab.bpe',
    )
    parser.add_argument(
        "--max-sent-index",
        type=int, default=120,
        help="Maximum number for <S#>",
    )
    parser.add_argument(
        "--data-format",
        default='switch',
        choices=['previous', 'reorder', 'switch'],
        help="Data format for different experiments",
    )
    parser.add_argument(
        "--source",
        default='exp_test/cnndm.tokenized/valid.source',
        help="source file to filter/encode",
    )
    parser.add_argument(
        "--target",
        default='exp_test/cnndm.tokenized/valid.target',
        help="target file to filter/encode",
    )
    parser.add_argument(
        "--outdir-tok",
        default='exp_test/cnndm.segmented_switch',
        help="output directory to save the encoded source/target files",
    )
    parser.add_argument(
        "--outdir-id",
        default='exp_test/cnndm.encoded_switch',
        help="output directory to save the encoded source/target files",
    )
    parser.add_argument(
        "--abs-threshold",
        default=0.7,
        type=float,
        help="Using oracle match to rewrite if its average recall is above a threshold, otherwise generate.",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not path.exists(args.outdir_tok):
        os.mkdir(args.outdir_tok)

    if not path.exists(args.outdir_id):
        os.mkdir(args.outdir_id)

    with contextlib.ExitStack() as stack:
        inputs = [stack.enter_context(open(args.source, "r", encoding="utf-8")),
                  stack.enter_context(open(args.target, "r", encoding="utf-8"))]
        outputs = [stack.enter_context(open(path.join(args.outdir_id, path.basename(args.source)), "w", encoding="utf-8")),
                   stack.enter_context(open(path.join(args.outdir_id, path.basename(args.target)), "w", encoding="utf-8")),
                   stack.enter_context(open(path.join(args.outdir_tok, path.basename(args.source)), "w", encoding="utf-8")),
                   stack.enter_context(open(path.join(args.outdir_tok, path.basename(args.target)), "w", encoding="utf-8"))
                   ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(tqdm(encoded_lines), start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

if __name__ == "__main__":
    main()
