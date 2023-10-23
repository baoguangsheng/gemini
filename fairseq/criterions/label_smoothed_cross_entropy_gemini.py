# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion

''' Common terms that we use:
      wtok - word token
      itok - identifier token
'''
def label_smoothed_nll_loss(lprobs, target, epsilon, itok_range=None, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    correct = (lprobs.argmax(dim=-1, keepdim=True) == target).int()
    nll_loss = -lprobs.gather(dim=-1, index=target)

    if nll_loss.max().item() > 100:
        print('WARNING: nll_loss is over 100!')

    smooth_loss = -(lprobs * (lprobs > -1e8).float()).sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        correct.masked_fill_(pad_mask, 0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    # Guangsheng Bao: log the separate losses on word and identifier tokens
    if itok_range is not None:
        itok_mask = (target >= itok_range[0]).float() * (target <= itok_range[1]).float()
        ntokens_itok = itok_mask.sum(dim=-1)
        nll_loss_itok = nll_loss * itok_mask
        nll_loss_wtok = nll_loss * (1 - itok_mask)
        correct_itok = correct * itok_mask
        correct_wtok = correct * (1 - itok_mask)
    else:
        ntokens_itok = torch.zeros_like(target)
        nll_loss_itok = torch.zeros_like(nll_loss)
        nll_loss_wtok = torch.zeros_like(nll_loss)
        correct_itok = torch.zeros_like(correct)
        correct_wtok = torch.zeros_like(correct)

    if reduce:
        ntokens_itok = ntokens_itok.sum()
        nll_loss_itok = nll_loss_itok.sum()
        nll_loss_wtok = nll_loss_wtok.sum()
        # nll_loss = nll_loss.sum()
        nll_loss = nll_loss_wtok * 0.9 + nll_loss_itok
        smooth_loss = smooth_loss.sum()
        correct_itok = correct_itok.sum()
        correct_wtok = correct_wtok.sum()

    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss_wtok, nll_loss_itok, correct_wtok, correct_itok, ntokens_itok

@register_criterion("label_smoothed_cross_entropy_gemini")
class GeminiLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss_wtok, nll_loss_itok, correct_wtok, correct_itok, ntokens_itok = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            'nll_loss_wtok': nll_loss_wtok.data,
            'nll_loss_itok': nll_loss_itok.data,
            'correct_wtok': correct_wtok.data,
            'correct_itok': correct_itok.data,
            'ntokens_itok': ntokens_itok.data,
            'ntokens': sample['ntokens'],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if net_output[1]['cross_ent']:
            cross_ent_token = net_output[1]['cross_ent_token']
            cross_ent_sent = net_output[1]['cross_ent_sent']
            logging_cross_ent = {
                "cross_ent": 1,
                "ntokens_ext": cross_ent_token[0][0].data,  # cnt_ext
                "ent_tok_ext": cross_ent_token[0][1].data,  # ent_ext
                "ntokens_abs": cross_ent_token[1][0].data,  # cnt_abs
                "ent_tok_abs": cross_ent_token[1][1].data,  # ent_abs
                "nsents_ext": cross_ent_sent[0][0].data,  # cnt_ext
                "ent_sent_ext": cross_ent_sent[0][1].data,  # ent_ext
                "nsents_abs": cross_ent_sent[1][0].data,  # cnt_abs
                "ent_sent_abs": cross_ent_sent[1][1].data,  # ent_abs
            }
            logging_output.update(logging_cross_ent)
        else:
            logging_output["cross_ent"] = 0
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        itok_range = (self.task.target_dictionary.index('<S>'),
                     self.task.target_dictionary.index('</S>'))
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss_wtok, nll_loss_itok, correct_wtok, correct_itok, ntokens_itok = label_smoothed_nll_loss(
            lprobs, target, self.eps, itok_range=itok_range, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss_wtok, nll_loss_itok, correct_wtok, correct_itok, ntokens_itok

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss_wtok"].avg))

        # Guangsheng Bao: log losses on word tokens and identifier tokens
        nll_loss_wtok_sum = sum(log.get('nll_loss_wtok', 0).sum() for log in logging_outputs)
        nll_loss_itok_sum = sum(log.get('nll_loss_itok', 0).sum() for log in logging_outputs)
        correct_wtok_sum = sum(log.get('correct_wtok', 0).sum() for log in logging_outputs)
        correct_itok_sum = sum(log.get('correct_itok', 0).sum() for log in logging_outputs)
        ntokens_itok = sum(log.get('ntokens_itok', 0).sum() for log in logging_outputs)

        metrics.log_scalar('nll_loss_wtok', nll_loss_wtok_sum / (ntokens - ntokens_itok) / math.log(2), (ntokens - ntokens_itok), round=3)
        metrics.log_scalar('nll_loss_itok', nll_loss_itok_sum / ntokens_itok / math.log(2), ntokens_itok, round=3)
        metrics.log_scalar('accuracy_wtok', correct_wtok_sum / (ntokens - ntokens_itok), (ntokens - ntokens_itok), round=3)
        metrics.log_scalar('accuracy_itok', correct_itok_sum / ntokens_itok, ntokens_itok, round=3)

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
