#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from operator import itemgetter
import numpy as np
import torch

from fairseq import data, options, tasks, utils, tokenizer
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_scorer import SequenceScorer

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

def main(args):
    assert args.path is not None, '--path required for evaluation!'

    args.tokens_per_sample = getattr(args, 'tokens_per_sample', 1024)
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    
    # Load dataset splits
    task = tasks.setup_task(args)
    print(tasks)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task)

    # Optimize ensemble for generation
    # for i, model in enumerate(models):
    #    models[i].make_generation_fast_(beamable_mm_beam_size=None if args.no_beamable_mm else args.beam)
    #    if args.fp16:
    #        models[i].half()

    print('| [eval] evaluate from file')
    eval_from_file(models, task, args, use_cuda)


def eval_from_file(models, task, args, use_cuda, source_filename=None,
                   target_filename=None, score_filename=None):
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # I/O files
    source_filename = source_filename if source_filename is not None else args.source_file
    target_filename = target_filename if target_filename is not None else args.target_file
    score_filename = score_filename if score_filename is not None else args.score_file
    if score_filename is None:
        score_filename = target_filename + ".eval.score"
    outfile = open(score_filename, "w")

    # Get sorted input (and reversed)
    sorted_inputs, sorted_keys, sorted_targets = _get_sorted_inputs(
        source_filename, args.num_shards, args.delimiter, target_filename, args.shard_id,
        args.dup_src, args.dup_tgt)

    # Build input iterator
    src_tokens = [
        tokenizer.Tokenizer.tokenize(src_str, src_dict, add_if_not_exist=False).long()
        for src_str in sorted_inputs]
    tgt_tokens = [
        tokenizer.Tokenizer.tokenize(tgt_str, tgt_dict, add_if_not_exist=False).long()
        for tgt_str in sorted_targets] if sorted_targets is not None else None
    src_sizes = np.array([t.numel() for t in src_tokens])
    tgt_sizes = np.array([t.numel() for t in tgt_tokens])
    print('| loading {} examples, {} tokens'.format(len(sorted_inputs), sum(src_sizes)))

    dataset = data.LanguagePairDataset(
        src_tokens, src_sizes, src_dict, tgt_tokens, tgt_sizes, tgt_dict, shuffle=False)
    itr = data.EpochBatchIterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=models[0].max_positions(),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    gen_timer = StopwatchMeter()
    scorer = SequenceScorer(models, task.target_dictionary)
    if use_cuda:
        scorer.cuda()

    all_scores = dict()
    score_sum = 0.
    count, sen_count = 0, 0
    results = scorer.score_batched_itr(itr, cuda=use_cuda, timer=gen_timer)
    wps_meter = TimeMeter()
    for sample_id, src_tokens, target_tokens, hypos in results:
        for i, hypo in enumerate(hypos):
            pos_scores = hypo['positional_scores']
            inf_scores = pos_scores.eq(float('inf')) | pos_scores.eq(float('-inf'))
            if inf_scores.any():
                print('| Skipping tokens with inf scores:',
                      task.target_dictionary.string(hypo['tokens'][inf_scores.nonzero()]))
                pos_scores = pos_scores[(~inf_scores).nonzero()]
            score_sum += pos_scores.sum()
            count += pos_scores.numel()
            sentence_score = hypo['score']
            if i == 0:
                all_scores[sample_id.tolist()] = sentence_score
        sen_count += 1
        wps_meter.update(src_tokens.size(0))

    print("| [eval] writing scores into {}".format(score_filename))
    # print(sids)
    for index in range(len(sorted_inputs)):
        outfile.write("{}{}".format(all_scores[sorted_keys[index]], args.delimiter))
    outfile.close()

    avg_nll_loss = -score_sum / count
    print('| Evaluated {} tokens in {:.1f}s ({:.2f} tokens/s)'.format(gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))
    print('| Loss: {:.4f}, Perplexity: {:.2f}'.format(avg_nll_loss, np.exp(avg_nll_loss)))


def _get_sorted_inputs(filename, num_shards=1, delimiter="\n",
                       targets_filename=None, worker_id=None, 
                       dup_src=1, dup_tgt=1):
    print("| getting sorted inputs")
    # read file and sort inputs according them according to input length.
    if num_shards > 1:
        assert worker_id == None
        source_filename = filename + ("%.2d" % worker_id)
    else:
        source_filename = filename
    print("| [src] {}".format(source_filename))

    with open(source_filename, "r", encoding="utf8") as f:
        text = f.read()
        records = text.split(delimiter)
        if records[-1].strip() == "":
            records.pop()
        inputs = [record.strip() for record in records for _ in range(dup_src)]
    
    if targets_filename is not None:
        if num_shards > 1:
            targets_filename += "%.2d" % worker_id
        with open(targets_filename, "r", encoding="utf8") as f:
            text = f.read()
            records = text.split(delimiter)
            if records[-1].strip() == "":
                records.pop()
            targets = [record.strip() for record in records for _ in range(dup_tgt)]
           
        assert len(targets) == len(inputs)
        print("| [trg] {}".format(targets_filename))

    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=itemgetter(1), reverse=True)
    # We'll need the keys to rearrange the inputs back into their original order
    sorted_keys = {}
    sorted_inputs = []
    sorted_targets = None if targets_filename is None else []
    for new_index, (orig_index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[orig_index])
        if targets_filename is not None:
            sorted_targets.append(targets[orig_index])
        sorted_keys[orig_index] = new_index
    return sorted_inputs, sorted_keys, sorted_targets


if __name__ == '__main__':
    parser = options.get_eval_mt_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
