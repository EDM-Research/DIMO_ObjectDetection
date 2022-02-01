from data import utils
from data import mrcnn_dimo
from training import mrcnn

DIMO_PATH = "D:/Datasets/DIMO/dimo"


def train_subsets(subsets):
    train, val = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets)

    print(f"training images: {len(train.image_ids)}")
    print(f"validation images: {len(val.image_ids)}")

    mrcnn.train(train, val, mrcnn_dimo.DimoConfig(), False)


def prepare_subsets(subsets):
    utils.create_dimo_masks(DIMO_PATH, subsets)
    utils.create_dimo_train_split(DIMO_PATH, subsets, seed=10)


if __name__ == "__main__":
    train_subsets(["real_jaigo_000-150"])
