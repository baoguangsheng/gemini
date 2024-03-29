#!/usr/bin/env python
import torch
from fairseq.models.bart import BARTModel
import argparse
import os.path as path
from fairseq.data.encoders.gpt2_bpe import get_encoder
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from extractor import build_extractor
from tqdm import tqdm

class BartRewriter:
    def __init__(self, args):
        self.args = args
        self.args.device = 'cuda'
        self.args.source_lang = 'source'
        self.args.target_lang = 'target'
        self.bpe = get_encoder(path.join(args.rewriter_path, 'encoder.json'), path.join(args.rewriter_path, 'vocab.bpe'))
        self.task = tasks.setup_task(self.args)
        self.model, self.model_args = checkpoint_utils.load_model_ensemble(
            [path.join(args.rewriter_path, 'checkpoint_best.pt')], task=self.task)
        self.model = self.model[0]
        self.model.make_generation_fast_(beamable_mm_beam_size=args.beam)
        self.model.cuda().eval()
        self.generator = self.task.build_generator([self.model], args)
        assert args.data_format == self.model_args.data_format

    def encode(self, line):
        line = ' ' + line.strip()
        ids = self.bpe.encode(line)
        return list(map(str, ids))

    def encode_lines(self, doc_sents, ext_idx):
        slines = doc_sents
        abs_art_idx = ext_idx
        bos = '<S%s>'
        eos = '</S>'

        # following code are copied from data_builder.py
        sids = []
        stokens = []
        if self.args.data_format == 'previous': # the format required by previous BERT ContextRewriter
            # source
            for idx, line in enumerate(slines):
                prefix = [bos % (tidx + 1) for tidx, sidx in enumerate(abs_art_idx) if sidx == idx]
                suffix = [eos] * len(prefix)
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
        elif self.args.data_format == 'reorder': # the format required by order sensitive rewriter
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
        elif self.args.data_format == 'switch':  # the format required by rewriter-generator
            # source
            tok_gen = bos % ''
            sids.append(tok_gen)
            stokens.append(tok_gen)
            for idx, line in enumerate(slines,):
                prefix = [bos % (idx + 1)] if idx < self.args.max_sent_index else []
                suffix = [eos] if idx < self.args.max_sent_index else []
                sids.extend(prefix + self.encode(line) + suffix)
                stokens.extend(prefix + line.split() + suffix)
        else:
            raise Exception()
        return sids, stokens

    def rewrite(self, docs, exts):
        assert len(docs) == len(exts)
        src_dict = self.model.encoder.dictionary
        tgt_dict = self.model.decoder.dictionary
        # make a batch
        src_tokens = []
        src_lengths = []
        oracle_exts = []
        for doc_sents, ext_idx in zip(docs, exts):
            sids, stokens = self.encode_lines(doc_sents, ext_idx)
            tokens = [src_dict.bos_index] + [src_dict.index(id) for id in sids]
            tokens = tokens[:self.model_args.max_source_positions - 1] + [src_dict.eos_index]
            src_tokens.append(tokens)
            src_lengths.append(len(tokens))
            oracle_exts.append(ext_idx)
        max_len = max(src_lengths)
        src_tokens = [line + [src_dict.pad_index] * (max_len - len(line)) for line in src_tokens]
        src_tokens = torch.tensor(src_tokens, device=self.args.device)
        src_lengths = torch.tensor(src_lengths, device=self.args.device)
        sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths},
                  'oracle_exts': oracle_exts}
        # generate
        hypos = self.generator(sample, prefix_tokens=None)
        hypos = [h[0]['tokens'].cpu().numpy().tolist() for h in hypos]
        assert len(hypos) == len(docs)
        sums = []
        newexts = []
        for hypo, ext_idx in zip(hypos, exts):
            sum = []
            ext = []
            for tok in hypo:
                tok = tgt_dict[tok]
                if tok in ['<s>', '</s>', '<pad>', '<unk>', '</S>']:
                    continue
                elif tok.startswith('<S') and tok.endswith('>'):
                    sum.append([])  # start a new sentence
                    ext.append(int(tok[2:-1]) - 1 if len(tok) > 3 else -1)
                else:
                    if len(sum) == 0:
                        sum.append([])
                        ext.append(-1)
                    sum[-1].append(int(tok))
            sum = [self.bpe.decode(sent).strip() for sent in sum if len(sent) > 0]
            sums.append(sum)
            # newexts.append(ext if len(ext_idx) == 0 else ext_idx)
            newexts.append(ext)
        return sums, newexts


def main():
    parser = options.get_generation_parser()
    parser.add_argument('--data-format', default='switch', choices=['previous', 'reorder', 'switch'],
                        help='data format for different experiments')
    parser.add_argument("--source", default="exp_switch/cnndm.tokenized/test.source", help="text to summarize")
    parser.add_argument("--target", default="exp_switch/cnndm.tokenized/test.target", help="golden summary")
    parser.add_argument("--extractor", default="none", choices=['none', 'oracle', 'lead3', 'bertsumext', 'bertext', 'switch'],
                        help="extractor for testing the rewriting.")
    parser.add_argument("--extractor-path", default="scripts_sum/bert_extractors/models",
                        help="where the extractor model is saved")
    parser.add_argument("--abs-threshold", default=0.7, type=float,
        help="Using oracle match to rewrite if its average recall is above a threshold, otherwise generate.",
    )
    parser.add_argument("--rewriter-path", default="exp_switch/run-switch/cnndm.checkpoints",
                        help="where the rewriter model is saved")
    parser.add_argument("--max-sent-index", type=int, default=120, help="Maximum number for <S#>",)
    parser.add_argument("--abspen", type=float, default=1.0, help="Penalty to abs mode", )
    parser.add_argument("--oracle-ext", default='none', choices=['none', 'len', 'mode', 'ext'],
                        help="oracle extraction", )
    parser.add_argument("--outdir", default="exp_switch/run-switch/cnndm.results", help="generated summary")
    args = options.parse_args_and_arch(parser)

    # prepare data
    with open(args.source, 'r') as fsrc, \
         open(args.target, 'r') as ftgt:
        slines = [line.strip() for line in fsrc]
        tlines = [line.strip() for line in ftgt]

    sep = ' <SEP> '
    batches = [([], [])]
    for sline, tline in zip(slines, tlines):
        batches[-1][0].append(sline.split(sep))
        batches[-1][1].append(tline.split(sep))
        if len(batches[-1][0]) == args.batch_size:
            batches.append(([], []))

    # prepare model
    extractor = None if args.extractor == 'none' else build_extractor(args)
    rewriter = BartRewriter(args)

    # summarize
    olines = []
    extlines = []
    idxlines = []
    for docs, refs in tqdm(batches):
        if len(docs) == 0:
            continue
        oracle_exts = [[]] * len(refs) if extractor is None else extractor.extract(docs, refs)
        sums, exts = rewriter.rewrite(docs, oracle_exts)
        ext_comp = list(zip(exts, oracle_exts))
        ext_comp = [e == o for e, o in ext_comp]
        if not all(ext_comp):
            sums, exts = rewriter.rewrite(docs, oracle_exts)
        extsums = [[doc[idx] for idx in ext if idx < len(doc) and idx >= 0] for doc, ext in zip(docs, exts)]
        assert len(sums) == len(docs)
        assert len(extsums) == len(docs)
        for ext in exts:
            idxlines.append(' '.join(map(str, ext)))
        for sum in extsums:
            extlines.append(sep.join(sum))
        for sum in sums:
            olines.append(sep.join(sum))
    assert len(olines) == len(slines)
    # write summaries
    split = path.split(args.source)[1].split('.')[0]
    with open(path.join(args.outdir, f'{split}.{args.extractor}.abspen{args.abspen}.gen'), 'w') as fout:
        fout.write('\n'.join(olines))
    with open(path.join(args.outdir, f'{split}.{args.extractor}.abspen{args.abspen}.ext'), 'w') as fout:
        fout.write('\n'.join(extlines))
    with open(path.join(args.outdir, f'{split}.{args.extractor}.abspen{args.abspen}.extidx'), 'w') as fout:
        fout.write('\n'.join(idxlines))

if __name__ == "__main__":
    main()
