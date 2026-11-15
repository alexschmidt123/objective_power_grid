"""One bounded process; imports alone never start training."""
import argparse
from runtime import *
from engine import run_training

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['diagnose','train'],required=True);p.add_argument('--seed',type=int);p.add_argument('--arm');args=p.parse_args()
 if args.task=='diagnose':
  from audit import frozen_diagnosis
  frozen_diagnosis()
 else:
  assert args.seed in [101,102,103,104,105] and args.arm in ['learned','uniform','v2_control','router_only','encoders_only']
  run_training(args.seed,args.arm,OUT/f'seed{args.seed}_{args.arm}')
if __name__=='__main__':main()
