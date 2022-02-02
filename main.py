from data import utils, mrcnn_dimo
from training import mrcnn as mrcnn_training

if __name__ == "__main__":
    utils.create_dimo_train_split("F:/Data/dimo", ["sim_jaigo", "real_jaigo"])
    train, val, config = mrcnn_dimo.get_dimo_datasets("F:/Data/dimo", ["real_jaigo"])
    mrcnn_training.train(train, val, config, True)