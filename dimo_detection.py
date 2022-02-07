import argparse
from main import prepare_subsets, train_subsets, show_subsets, test_subsets


parser = argparse.ArgumentParser()
parser.add_argument('action', type=str)
parser.add_argument('--subsets', required=True, type=str, nargs='+')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('-o', action='store_true')
parser.add_argument('-a', action='store_true')
parser.add_argument('-t', action='store_true')
args = parser.parse_args()

action = args.action

if action == 'prepare':
    override = args.o
    prepare_subsets(args.subsets, override)
elif action == 'show':
    show_subsets(args.subsets)
elif action == 'train':
    augment = args.a
    transfer_learning = args.t
    model_id = args.model
    train_subsets(args.subsets, augment=augment, transfer_learning=transfer_learning, model_id=model_id)
elif action == 'test':
    test_subsets(args.subsets, args.model)
else:
    parser.print_help()