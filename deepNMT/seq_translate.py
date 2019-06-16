from generate import main as single_model_main

import os, sys, subprocess
import time, sys
import re
from fairseq import options

def obtain_sys_argv():
    def if_illegal_args(str_check):
        illegal_args_names = ['--ckpt', '--initial']
        return any([x in str_check for x in illegal_args_names])
    sys_args = ' '.join([x for (idx, x) in enumerate(sys.argv[1:]) if not if_illegal_args(x) and not if_illegal_args(sys.argv[idx])])
    return sys_args

def main(args):
    print(args)
    files = [os.path.join(args.ckpt_dir, x) for x in os.listdir(args.ckpt_dir) if x.endswith('.pt')]
    files.sort(key=lambda x: os.path.getmtime(x))

    start_idx = 0
    if args.initial_model is not None:
        initial_model = os.path.join(args.ckpt_dir, args.initial_model)
        for (idx, file) in enumerate(files):
            if file == initial_model:
                start_idx = idx

    bleu_ptn = 'BLEU4\s=\s([\d\.]+?),'
    for x in range(start_idx, len(files)):
        ckpt_file = files[x]
        args.path = ckpt_file
        #Note here, simply calling single_model_main will bring mysterious memory error, so use bruteforce calling instead
        #single_model_main(args)
        pl_process = subprocess.Popen(
            'python /var/storage/shared/sdrgvc/fetia/fairseq/generate.py {} --path {}'.format(obtain_sys_argv(), ckpt_file),
            shell=True,
            stdout=subprocess.PIPE)
        pl_output = pl_process.stdout.read()
        bleu_match = re.search(bleu_ptn, str(pl_output))
        if bleu_match:
            bleu_score = bleu_match.group(1)
            print(ckpt_file, bleu_score)
            sys.stdout.flush()
        time.sleep(15)

if __name__ == '__main__':
    parser = options.get_generation_parser(seq=True)
    args = options.parse_args_and_arch(parser)
    main(args)