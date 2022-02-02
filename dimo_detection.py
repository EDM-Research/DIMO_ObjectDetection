import argparse
from main import prepare_subsets, train_subsets, show_subsets


parser = argparse.ArgumentParser()
parser.add_argument('action', type=str)
parser.add_argument('--subsets', required=True, type=str, nargs='+')
args = parser.parse_args()

action = args.action

if action == 'prepare':
    prepare_subsets(args.subsets)
elif action == 'show':
    show_subsets(args.subsets)
elif action == 'train':
    train_subsets(args.subsets)