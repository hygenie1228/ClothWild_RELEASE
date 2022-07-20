import os
from config import cfg
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True

    from base import Tester

    if args.type == 'cd': 
        cfg.calculate_cd = True
        cfg.testset = ['PW3D']
    elif args.type == 'bcc': 
        cfg.calculate_bcc = True
        cfg.testset = ['MSCOCO']
    else:
        assert 0, 'Test type is invalid.'

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):
        if  itr < cur_sample_idx:
            continue
        
        for k,v in inputs.items():
            if type(v) is torch.Tensor: inputs[k] = v.cuda()
        for k,v in targets.items():
            if type(v) is torch.Tensor: targets[k] = v.cuda()
        for k,v in meta_info.items():
            if type(v) is torch.Tensor: meta_info[k] = v.cuda()
        
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')

        # save output
        _out = {}
        for k,v in out.items():
            if type(v) is torch.Tensor:
                _out[k] = v.cpu().numpy()
                batch_size = v.shape[0]
            else:
                _out[k] = v
        out = _out
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]
        
        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k,v in cur_eval_result.items():
            if k in eval_result: eval_result[k] += v
            else: eval_result[k] = v
        cur_sample_idx += len(out)
    
    tester._print_eval_result(eval_result)

if __name__ == "__main__":
    main()
