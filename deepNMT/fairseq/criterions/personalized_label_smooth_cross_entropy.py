# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('personalized_label_smoothed_cross_entropy')
class PersonalizedLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.n_tokens = len(task.tgt_dict)
        self.contrib = torch.tensor([self.eps / (self.n_tokens - 1) * (self.n_tokens - i - 1) for i in range(self.n_tokens)]).cuda()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def forward(self, model, sample, reduce=True, teacher_outputs=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert reduce
        net_output = model(**sample['net_input']) #B x T x V
        lprobs = model.get_normalized_probs(net_output, log_probs=True) #B x T x V
        lprobs = lprobs.view(-1, lprobs.size(-1)) # BT x V
        assert lprobs.size(-1) == self.n_tokens

        target = model.get_targets(sample, net_output).view(-1, 1) #BT x 1, before view: B x T
        non_pad_mask = target.ne(self.padding_idx) # BT x 1

        trade_off = self.contrib[target.view(target.size(0))][non_pad_mask.view(non_pad_mask.size(0))]

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask] #BT
        nll_loss_sum = nll_loss.sum()
        nll_loss = torch.sum(nll_loss * (1. - trade_off))

        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        smooth_loss = torch.sum(smooth_loss * trade_off) / self.n_tokens

        loss = nll_loss + smooth_loss

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss_sum.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
