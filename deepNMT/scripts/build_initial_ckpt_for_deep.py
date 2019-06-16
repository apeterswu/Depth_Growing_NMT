#!/usr/bin/env python3

import sys
import torch


def load_parent_ckpt(ckpt):
    state = torch.load(
            ckpt,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
    return state
       
def main():
    parent = sys.argv[1]
    new_state = sys.argv[2]
    output_path = sys.argv[3]
    
    parent = load_parent_ckpt(parent)
    new_state = load_parent_ckpt(new_state)
    
    for (k,v) in parent['model'].items():
        new_state['model'][k] = v
    
    torch.save(new_state, output_path)
       

if __name__ == '__main__':
    main()
