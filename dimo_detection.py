import argparse
from main import prepare_subsets, train_subsets, show_subsets, test_subsets, test_batch


parser = argparse.ArgumentParser()
parser.add_argument('action', type=str)
parser.add_argument('--subsets', type=str, nargs='+')
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--image_count', type=int, default=None)
parser.add_argument('-o', action='store_true')
parser.add_argument('-a', action='store_true')
parser.add_argument('-t', action='store_true')
parser.add_argument('-s', action='store_true')
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

action = args.action

if action == 'prepare':
    override = args.o
    split_scenes = args.s
    prepare_subsets(args.subsets, override, split_scenes)
elif action == 'show':
    show_subsets(args.subsets)
elif action == 'train':
    augment = args.a
    train_image_count = args.image_count
    transfer_learning = args.t
    model_id = args.model
    train_subsets(args.subsets, augment=augment, transfer_learning=transfer_learning, model_id=model_id, train_image_count=train_image_count)
elif action == 'test':
    if args.file:
        test_batch(args.file)
    else:
        save = args.save
        test_subsets(args.subsets, args.model, save_results=save)
else:
    parser.print_help()