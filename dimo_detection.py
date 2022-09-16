import argparse


def main_program():
    from main import prepare_subsets, train_subsets, show_subsets, test_subsets, test_batch

    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str)
    parser.add_argument('--subsets', type=str, nargs='+')
    parser.add_argument('--ft_subsets', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dimo_path', type=str, default=None)
    parser.add_argument('--layers', type=str, default=None)
    parser.add_argument('--file', type=str, default=None)
    parser.add_argument('--image_counts', type=int, default=None, nargs='+')
    parser.add_argument('--ft_image_counts', type=int, default=None, nargs='+')
    parser.add_argument('-o', action='store_true')
    parser.add_argument('-a', action='store_true')
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-s', action='store_true')
    parser.add_argument('--save_all', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    action = args.action
    dimo_path = args.dimo_path

    if action == 'prepare':
        override = args.o
        split_scenes = args.s
        prepare_subsets(args.subsets, override, split_scenes, dimo_path=dimo_path)
    elif action == 'show':
        show_subsets(args.subsets, dimo_path=dimo_path)
    elif action == 'train':
        save_all = args.save_all
        augment = args.a
        train_image_counts = args.image_counts
        transfer_learning = args.t
        model_id = args.model
        layers = args.layers

        ft_subsets = args.ft_subsets
        ft_image_counts = args.ft_image_counts
        train_subsets(args.subsets, augment=augment, transfer_learning=transfer_learning, model_id=model_id,
                      train_image_counts=train_image_counts, ft_subsets=ft_subsets, ft_image_count=ft_image_counts, layers=layers, save_all=save_all, dimo_path=dimo_path)
    elif action == 'test':
        if args.file:
            test_batch(args.file)
        else:
            save = args.save
            test_subsets(args.subsets, args.model, save_results=save)
    else:
        parser.print_help()


if __name__ == "__main__":
    main_program()