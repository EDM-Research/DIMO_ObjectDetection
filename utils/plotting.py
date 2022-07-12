from matplotlib import pyplot as plt

import file_io


def plot_epochs(models: list, model_to_title: dict = None):
    with plt.style.context('Solarize_Light2'):
        for model_id in models:
            epochs, aps = file_io.read_model_epochs(model_id)
            plt.plot(epochs, aps, label=model_id if model_to_title is None else model_to_title[model_id])

        plt.legend()
        plt.show()
        plt.xlabel("Epoch")
        plt.ylabel("AP")